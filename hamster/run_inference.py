"""Run the released HAMSTER VLM, preserving native response and every waypoint."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from PIL import Image

from hamster.download import REPO, REVISION
from hamster.trajectory import postprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default="/workspace/models/Hamster_dev")
    parser.add_argument("--image", default="examples/long_task01_ep0001.png")
    parser.add_argument("--task", default="Grasp the gray cup in the foreground on the table, then lift it off the table.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upstream-user-only", action="store_true",
                        help="Reproduce upstream server's missing assistant generation prefix")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model_path = (Path(args.model_root) / "model_path.txt").read_text().strip()
    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    image.save(output / "input.png")
    prompt = Path(__file__).with_name("prompt.txt").read_text().format(task=args.task).rstrip("\n")
    (output / "prompt.txt").write_text(prompt)
    import torch
    from transformers import set_seed
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    set_seed(args.seed)
    disable_torch_init()
    metadata = {"model": REPO, "revision": REVISION, "base_vlm": "VILA-1.5-13B",
                "vila_revision": subprocess.check_output(["git", "-C", "external/VILA", "rev-parse", "HEAD"], text=True).strip(),
                "dtype": "float16", "attention": "sdpa", "quantization": None,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "image_size": list(image.size), "task": args.task, "seed": args.seed,
                "do_sample": False, "use_cache": True, "max_new_tokens": args.max_new_tokens,
                "prompt_source": "HAMSTER_beta/gradio_server_example.py at 526a37f",
                "conversation": "vicuna_v1; user-only as upstream server.py" if args.upstream_user_only else "vicuna_v1; assistant generation prefix",
                "gpu": torch.cuda.get_device_name(0),
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate", "Pillow")}}
    def save_metadata():
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    save_metadata()
    print("Loading HAMSTER", flush=True)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, "Hamster_dev", attn_implementation="sdpa")
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
    if not args.upstream_user_only:
        conv.append_message(conv.roles[1], None)
    formatted = conv.get_prompt()
    (output / "formatted_prompt.txt").write_text(formatted)
    images = process_images([image], processor, model.config).to(model.device, dtype=torch.float16)
    ids = tokenizer_image_token(formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    print("Generating HAMSTER trajectory", flush=True)
    with torch.inference_mode():
        generated = model.generate(ids, images=[images], do_sample=False,
                                   max_new_tokens=args.max_new_tokens, use_cache=True)
    torch.cuda.synchronize()
    metadata.update(generation_seconds=time.perf_counter() - start,
                    returned_token_ids=generated[0].tolist(), input_tokens=ids.shape[-1],
                    output_tokens=generated.shape[-1],
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    reached_token_limit=generated.shape[-1] >= args.max_new_tokens and generated[0, -1].item() != tokenizer.eos_token_id)
    save_metadata()
    (output / "raw_response.txt").write_text(tokenizer.decode(generated[0], skip_special_tokens=False))
    (output / "response.txt").write_text(tokenizer.decode(generated[0], skip_special_tokens=True))
    try:
        if metadata["reached_token_limit"]:
            raise ValueError("Generation reached token limit")
        result = postprocess(output)
        print(json.dumps(result, indent=2), flush=True)
    except ValueError as exc:
        (output / "validation_error.txt").write_text(str(exc))
        raise
    print(f"Results saved: {output}", flush=True)


if __name__ == "__main__":
    main()
