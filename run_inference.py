"""One image -> Qwen prediction -> validated JSON -> deterministic overlay."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from PIL import Image
from postprocess import postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", default="抓住桌面前景的灰色杯子，然后把它抬离桌面。")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    image_path = Path(args.image).resolve()
    image = Image.open(image_path).convert("RGB")
    prompt = (Path(__file__).parent / "prompts/trajectory.txt").read_text().format(
        task=args.task, width=image.width, height=image.height)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    image.save(output / "input.png")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
    set_seed(args.seed)
    metadata = {"model": args.model if not Path(args.model).exists() else Path(args.model).name,
                "revision": args.revision, "dtype": "bfloat16", "quantization": None,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "image_size": list(image.size), "task": args.task, "seed": args.seed,
                "thinking": args.thinking, "reasoning_effort": "medium", "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "gpu": torch.cuda.get_device_name(0),
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate", "Pillow")}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("Loading processor and BF16 model", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa").eval()
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=args.thinking, reasoning_effort="medium").to(model.device)
    (output / "formatted_prompt.txt").write_text(processor.tokenizer.decode(inputs.input_ids[0]), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    print("Generating trajectory", flush=True)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    new_ids = ids[0, inputs.input_ids.shape[-1]:]
    raw = processor.decode(new_ids, skip_special_tokens=False)
    text = processor.decode(new_ids, skip_special_tokens=True)
    (output / "raw_response.txt").write_text(raw, encoding="utf-8")
    (output / "response.txt").write_text(text, encoding="utf-8")
    metadata.update(generation_seconds=time.perf_counter()-start, input_tokens=inputs.input_ids.shape[-1],
                    output_tokens=len(new_ids), peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    reached_token_limit=len(new_ids) >= args.max_new_tokens)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    try:
        if metadata["reached_token_limit"]:
            raise ValueError("Generation reached token limit; raw response saved, not rendered")
        report = postprocess(output)
    except (ValueError, KeyError, TypeError) as exc:
        (output / "validation_error.txt").write_text(str(exc))
        raise
    print(json.dumps(report, indent=2), flush=True)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    print(f"Results saved to {output}", flush=True)


if __name__ == "__main__":
    main()
