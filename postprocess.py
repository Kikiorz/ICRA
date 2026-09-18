"""Render saved predictions, distinguishing drawable coordinates from prompt compliance."""
import argparse
import json
from pathlib import Path
from PIL import Image
from trajectory import parse_response, validate
from render import render


def postprocess(output):
    output = Path(output)
    text = (output / "response.txt").read_text()
    with Image.open(output / "input.png") as image:
        size = image.size
    data = parse_response(text)
    # Fatal geometry/schema errors are not repaired or rendered.
    validate(data, size, strict_gripper=False)
    report = {"drawable_structure_valid": True, "strict_prompt_schema_valid": True,
              "coordinates_modified": False, "physical_validity_evaluated": False, "errors": []}
    try:
        validate(data, size)
    except ValueError as exc:
        report["strict_prompt_schema_valid"] = False
        report["errors"].append(str(exc))
    (output / "schema_validation.json").write_text(json.dumps(report, indent=2))
    (output / "trajectory.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    render(output / "input.png", data, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(postprocess(args.output_dir), indent=2))
