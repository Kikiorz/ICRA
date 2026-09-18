"""Parse HAMSTER's native output without executing model-generated code."""
import ast
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


def parse_response(text):
    answers = re.findall(r"<ans>(.*?)</ans>", text, re.DOTALL)
    if len(answers) != 1:
        raise ValueError("Expected exactly one complete <ans> block")
    body = answers[0]
    for action in ("Open Gripper", "Close Gripper"):
        body = body.replace(f"<action>{action}</action>", repr(action))
    try:
        entries = ast.literal_eval(body)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Invalid trajectory literal") from exc
    if not isinstance(entries, list):
        raise ValueError("Trajectory must be a list")
    points, events = [], []
    for entry in entries:
        if isinstance(entry, str):
            if entry not in ("Open Gripper", "Close Gripper") or not points:
                raise ValueError("Action must be known and follow a waypoint")
            events.append({"after_step": len(points) - 1, "action": entry})
        else:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("Each point must contain exactly x and y")
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in entry):
                raise ValueError("Coordinates must be finite numbers in [0, 1]")
            points.append({"step": len(points), "x": entry[0], "y": entry[1]})
    if len(points) < 2:
        raise ValueError("Expected at least two waypoints")
    return {"coordinate_system": "normalized_1", "tracked_point": "gripper_end_effector",
            "waypoints": points, "gripper_events": events}


def postprocess(directory):
    directory = Path(directory)
    result = parse_response((directory / "response.txt").read_text())
    image = Image.open(directory / "input.png").convert("RGB")
    result["image_size"] = list(image.size)
    pixels = [(p["x"] * (image.width - 1), p["y"] * (image.height - 1))
              for p in result["waypoints"]]
    # Pixel-centre convention matches the existing Qwen renderer. No interpolation,
    # smoothing, stage inference, or initial gripper-state assumption is applied.
    draw = ImageDraw.Draw(image)
    draw.line(pixels, fill="#ffd34e", width=2)
    for i, (x, y) in enumerate(pixels):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#00c7ff", outline="black")
        draw.text((x + 4, y - 9), str(i), fill="white", stroke_width=1, stroke_fill="black")
    image.save(directory / "overlay.png")
    large = image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (large.width + 400, max(large.height, 400)), "#17212b")
    canvas.paste(large, (0, 0))
    font = ImageFont.load_default(size=17)
    lines = ["HAMSTER / VILA-1.5-13B", "Model-predicted waypoints", ""]
    events = result["gripper_events"]
    for point, (x, y) in zip(result["waypoints"], pixels):
        lines.append(f"{point['step']}: ({x:.1f}, {y:.1f}) px")
        lines.extend("  " + e["action"] for e in events if e["after_step"] == point["step"])
    lines += ["", "2D prediction only", "No simulated execution"]
    ImageDraw.Draw(canvas).multiline_text((large.width + 15, 20), "\n".join(lines),
                                         fill="white", font=font, spacing=7)
    canvas.save(directory / "trajectory.png")
    (directory / "trajectory.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "pixel_waypoints.json").write_text(json.dumps(pixels, indent=2) + "\n")
    return result
