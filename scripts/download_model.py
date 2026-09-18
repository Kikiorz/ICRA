"""Download the exact model revision used for this experiment."""
import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
snapshot_download("Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                  local_dir=args.output, max_workers=8,
                  ignore_patterns=["*.md", "*.pdf", "*.png", "*.jpg"])
