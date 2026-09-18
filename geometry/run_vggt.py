"""Single-image VGGT prediction, with raw tensors and unscaled colored PLY."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from huggingface_hub import snapshot_download

MODEL = "facebook/VGGT-1B"
REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"


def write_ply(path, points, colors, confidence, valid):
    height, width = confidence.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                      ('confidence', '<f4'), ('pixel_u', '<u2'), ('pixel_v', '<u2')])
    vertices = np.empty(int(valid.sum()), dtype=dtype)
    for i, key in enumerate(('x', 'y', 'z')):
        vertices[key] = points[..., i][valid]
    for i, key in enumerate(('red', 'green', 'blue')):
        vertices[key] = colors[..., i][valid]
    vertices['confidence'] = confidence[valid]
    vertices['pixel_u'], vertices['pixel_v'] = u[valid], v[valid]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment VGGT estimated geometry; scale is NOT calibrated in meters\n'
              f'element vertex {len(vertices)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'property float confidence\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n')
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        vertices.tofile(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='examples/long_task01_ep0001.png')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-dir', default='/workspace/models/VGGT-1B')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.image)
    image = Image.open(source).convert('RGB')
    image.save(output / 'input.png')
    metadata = dict(model=MODEL, revision=REVISION,
        code_revision=subprocess.check_output(['git', '-C', 'external/vggt', 'rev-parse', 'HEAD'], text=True).strip(),
        image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), image_size=list(image.size),
        num_views=1, metric_scale_calibrated=False, known_camera_intrinsics=False,
        coordinate_frame='VGGT world; extrinsic maps world to OpenCV camera (x right, y down, z forward)',
        point_source='predicted depth unprojected using predicted camera intrinsics and extrinsics',
        preprocessing='official crop mode: width 518, aspect preserved with height rounded to multiple of 14; center-crop height above 518',
        ply_storage='float32 xyz; uint8 RGB; float32 confidence; uint16 processed-image pixel indices',
        filtering='only nonfinite points/confidence and nonpositive depth removed; no confidence threshold',
        autocast_dtype='bfloat16', weights_dtype='float32', seed=42,
        versions={p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'numpy', 'Pillow', 'huggingface-hub', 'einops']})
    def save_metadata():
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_metadata()
    print('Downloading pinned VGGT weights', flush=True)
    snapshot_download(MODEL, revision=REVISION, local_dir=args.model_dir,
                      allow_patterns=['config.json', 'model.safetensors'])
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    torch.manual_seed(42)
    model = VGGT.from_pretrained(args.model_dir).to('cuda').eval()
    images = load_and_preprocess_images([str(source)], mode='crop').to('cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    print('Predicting geometry', flush=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(images)
    torch.cuda.synchronize()
    metadata['inference_seconds'] = time.perf_counter() - start
    metadata['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred['pose_enc'], images.shape[-2:])
    arrays = {key: pred[key][0].float().cpu().numpy() for key in
              ['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf']}
    arrays['extrinsic'] = extrinsic[0].float().cpu().numpy()
    arrays['intrinsic'] = intrinsic[0].float().cpu().numpy()
    arrays['rgb'] = np.rint(images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    arrays['unprojected_points'] = unproject_depth_map_to_point_map(arrays['depth'], arrays['extrinsic'], arrays['intrinsic'])
    np.savez_compressed(output / 'predictions.npz', **arrays)
    depth, conf = arrays['depth'][0, ..., 0], arrays['depth_conf'][0]
    points = arrays['unprojected_points'][0]
    valid = np.isfinite(points).all(-1) & np.isfinite(conf) & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError('No valid predicted points')
    write_ply(output / 'pointcloud.ply', points, arrays['rgb'], conf, valid)
    Image.fromarray(arrays['rgb']).save(output / 'model_input.png')
    # Only the depth preview is contrast-normalized. Raw geometry remains unchanged.
    low, high = np.percentile(depth[valid], [2, 98])
    preview = np.clip((depth - low) / max(float(high - low), 1e-8), 0, 1)
    # Near: turquoise, far: dark blue. The scale is arbitrary, not meters.
    near, far = np.array([67, 226, 192]), np.array([23, 31, 85])
    color = np.rint(near[None, None, :] * (1-preview[..., None]) + far[None, None, :] * preview[..., None]).astype(np.uint8)
    color[~valid] = 0
    Image.fromarray(color).save(output / 'depth_preview.png')
    height, width = depth.shape
    metadata.update(processed_size=[width, height], point_count=int(valid.sum()),
                    rejected_points=int(valid.size-valid.sum()),
                    depth_percentiles=np.percentile(depth[valid], [0, 2, 50, 98, 100]).tolist(),
                    confidence_percentiles=np.percentile(conf[valid], np.arange(101)).tolist(),
                    confidence_is_probability=False)
    save_metadata()
    camera = dict(intrinsic=arrays['intrinsic'][0].tolist(), extrinsic=arrays['extrinsic'][0].tolist(),
                  image_size=[width, height], convention='OpenCV camera from world; uncalibrated scale')
    (output / 'camera.json').write_text(json.dumps(camera, indent=2) + '\n')
    print(f"Saved {metadata['point_count']} points to {output}", flush=True)


if __name__ == '__main__':
    main()
