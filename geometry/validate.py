"""Verify export integrity and reprojection, not physical reconstruction accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate(directory, source):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text())
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == meta['image_sha256']
    raw = np.load(directory / 'predictions.npz', allow_pickle=False)
    assert np.array_equal(np.array(Image.open(directory / 'model_input.png')), raw['rgb'])
    payload = (directory / 'pointcloud.ply').read_bytes()
    header, binary = payload.split(b'end_header\n', 1)
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                      ('confidence', '<f4'), ('u', '<u2'), ('v', '<u2')])
    vertices = np.frombuffer(binary, dtype=dtype)
    assert len(vertices) == meta['point_count']
    assert f'element vertex {len(vertices)}'.encode() in header
    u, v = vertices['u'], vertices['v']
    xyz = np.stack([vertices[k] for k in ('x', 'y', 'z')], axis=-1)
    rgb = np.stack([vertices[k] for k in ('r', 'g', 'b')], axis=-1)
    reference = raw['unprojected_points'][0, v, u]
    # PLY uses standard float32 coordinates; the upstream unprojection returns
    # float64. Verify the exact float32 cast and quantify that rounding alone.
    assert np.array_equal(xyz, reference.astype(np.float32))
    assert np.array_equal(rgb, raw['rgb'][v, u])
    assert np.array_equal(vertices['confidence'], raw['depth_conf'][0, v, u])
    assert np.isfinite(xyz).all()
    extrinsic, intrinsic = raw['extrinsic'][0], raw['intrinsic'][0]
    camera_points = xyz @ extrinsic[:, :3].T + extrinsic[:, 3]
    projected = camera_points @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:]
    expected = np.stack([u, v], axis=-1)
    error = np.linalg.norm(pixels - expected, axis=-1)
    assert error.max() < 1e-3, f'Reprojection error: {error.max()} pixels'
    assert np.allclose(camera_points[:, 2], raw['depth'][0, v, u, 0], atol=1e-5)
    return dict(points=len(vertices), max_reprojection_error_pixels=float(error.max()),
                coordinates_match_float32_cast=True,
                max_ply_rounding_error=float(np.max(np.abs(xyz-reference))), colors_preserved=True,
                physical_accuracy_evaluated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--source', default='examples/long_task01_ep0001.png')
    args = parser.parse_args()
    report = validate(args.directory, args.source)
    print(json.dumps(report, indent=2))
