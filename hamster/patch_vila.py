"""Make unused vision encoders lazy imports; preserve HAMSTER's SigLIP path."""
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vila", type=Path)
    args = parser.parse_args()
    path = args.vila / "llava/model/multimodal_encoder/builder.py"
    content = path.read_text()
    for module, cls, branch in [
        ("intern_encoder", "InternVisionTower", '    if "intern" in vision_tower_name.lower():\n'),
        ("radio_encoder", "RADIOVisionTower", '    elif "radio" in vision_tower_name:\n'),
    ]:
        line = f"from .{module} import {cls}\n"
        if line in content and not f"        {line}" in content:
            if branch not in content:
                raise ValueError(f"Unknown VILA source: missing {branch!r}")
            content = content.replace(line, "", 1).replace(branch, branch + "        " + line, 1)
    path.write_text(content)
    # Sequence-parallel training is unused in single-GPU inference. Import
    # DeepSpeed only inside the two entry points that actually use it.
    path = args.vila / "llava/train/sequence_parallel/globals.py"
    content = path.read_text()
    if "\nimport deepspeed.comm as dist\n" in content:
        content = content.replace("\nimport deepspeed.comm as dist\n", "\n", 1)
        for signature, indent in [
            ("    def __init__(self, ulysses_degree, ring_degree, dp_degree, use_ulysses_low, ring_type):\n", "        "),
            ("def set_pg_manager(sp_degree, sp_ring_degree=1, use_ulysses_low=True, ring_type=None):\n", "    "),
        ]:
            if signature not in content:
                raise ValueError("Unknown VILA sequence parallel source")
            content = content.replace(signature, signature + indent + "import deepspeed.comm as dist\n", 1)
        path.write_text(content)
