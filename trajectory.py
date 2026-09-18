"""Strict validation and coordinate conversion; never repair a model trajectory."""
import json
import math


def parse_response(text):
    # Thinking is preserved in raw_response.txt; only the final answer is parsed.
    answer = text.rsplit("</think>", 1)[-1].strip()
    if answer.startswith("```"):
        lines = answer.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Unclosed JSON code fence")
        answer = "\n".join(lines[1:-1])
    return json.loads(answer)


def coordinate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Coordinates must be numbers")
    if not math.isfinite(value) or not 0 <= value <= 1000:
        raise ValueError("Coordinates must be finite and in [0, 1000]")
    return value


def validate(data, image_size, *, strict_gripper=True):
    if data.get("coordinate_system") != "normalized_1000":
        raise ValueError("Unexpected coordinate system")
    if data.get("tracked_point") != "gripper_tcp":
        raise ValueError("Expected gripper_tcp")
    if data.get("image_size") != list(image_size):
        raise ValueError("Image dimensions do not match the source")
    for key in ("task", "grasp_strategy"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Missing {key}")
    assumptions = data.get("assumptions")
    if not isinstance(assumptions, list) or not all(isinstance(s, str) for s in assumptions):
        raise ValueError("assumptions must be a list of strings")
    target = data.get("target", {})
    if not isinstance(target.get("description"), str):
        raise ValueError("Missing target description")
    bbox = target.get("bbox", [])
    if len(bbox) != 4:
        raise ValueError("Target bbox must have four coordinates")
    x0, y0, x1, y1 = map(coordinate, bbox)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Target bbox is reversed or empty")
    points = data.get("waypoints", [])
    if not 8 <= len(points) <= 12:
        raise ValueError("Expected 8–12 waypoints")
    phases = {"start": 0, "approach": 1, "grasp": 2, "lift": 3}
    grip = {"start": "open", "approach": "open", "grasp": "close", "lift": "closed"}
    order = []
    for i, point in enumerate(points):
        if type(point.get("step")) is not int or point["step"] != i:
            raise ValueError("Steps must be consecutive integers starting at zero")
        coordinate(point["x"])
        coordinate(point["y"])
        phase = point.get("phase")
        if phase not in phases or point.get("gripper") not in {"open", "close", "closed"}:
            raise ValueError("Invalid phase or gripper state")
        if strict_gripper and point["gripper"] != grip[phase]:
            raise ValueError(f"Step {i}: phase {phase} expects gripper={grip[phase]}, got {point['gripper']}")
        order.append(phases[phase])
    if order != sorted(order) or set(order) != set(phases.values()) or order.count(0) != 1:
        raise ValueError("Expected one start, then approach, grasp, lift")
    return data


def to_pixel(x, y, image_size):
    width, height = image_size
    return coordinate(x) * (width - 1) / 1000, coordinate(y) * (height - 1) / 1000
