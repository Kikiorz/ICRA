"""Download only the released HAMSTER model components at a pinned revision."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

REPO = "yili18/Hamster_dev"
REVISION = "794f1f925c87e861d2f562943e978cc11f8c344d"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/workspace/models/Hamster_dev")
    args = parser.parse_args()
    names = HfApi().list_repo_files(REPO, revision=REVISION)
    roots = [name.removesuffix("/config.json") for name in names
             if name.endswith("/config.json") and name.count("/") == 1]
    if len(roots) != 1:
        raise ValueError(f"Expected one model root, got {roots}")
    root = roots[0]
    snapshot_download(REPO, revision=REVISION, local_dir=args.output,
                      allow_patterns=[f"{root}/config.json", f"{root}/llm/*",
                                      f"{root}/vision_tower/*", f"{root}/mm_projector/*"],
                      max_workers=4)
    path = Path(args.output) / root
    (Path(args.output) / "model_path.txt").write_text(str(path.resolve()) + "\n")
    print(f"HAMSTER model ready: {path}", flush=True)
