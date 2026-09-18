"""Overlay model coordinates with Pillow; no image generation or path correction."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from trajectory import to_pixel, validate

COLORS = {"start": "#ffffff", "approach": "#00bfff", "grasp": "#ffcb33", "lift": "#fa6575"}


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def render(image_path, data, output_dir):
    source = Image.open(image_path).convert("RGB")
    # State/phase mismatches are reported separately; they do not prevent displaying
    # finite, correctly ordered model coordinates. No field is changed here.
    validate(data, source.size, strict_gripper=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pixel_points = [{**p, "pixel_x": to_pixel(p["x"], p["y"], source.size)[0],
                     "pixel_y": to_pixel(p["x"], p["y"], source.size)[1]}
                    for p in data["waypoints"]]
    (output_dir / "pixel_waypoints.json").write_text(json.dumps(pixel_points, indent=2), encoding="utf-8")
    for scale, name in ((1, "overlay.png"), (4, "trajectory.png")):
        base = source.resize((source.width * scale, source.height * scale), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(base)
        xy = [(p["pixel_x"] * scale, p["pixel_y"] * scale) for p in pixel_points]
        for i in range(1, len(xy)):
            a, b = xy[i - 1], xy[i]
            color = COLORS[pixel_points[i]["phase"]]
            draw.line([a, b], fill=color, width=max(1, scale))
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length > 5 * scale:
                ux, uy = dx / length, dy / length
                tip = (a[0] + dx * 0.65, a[1] + dy * 0.65)
                size = 2.5 * scale
                draw.polygon([tip, (tip[0] - size*ux - size*uy*.55, tip[1] - size*uy + size*ux*.55),
                              (tip[0] - size*ux + size*uy*.55, tip[1] - size*uy - size*ux*.55)], fill=color)
        for i, (x, y) in enumerate(xy):
            r = 2 * scale
            draw.ellipse((x-r, y-r, x+r, y+r), fill=COLORS[pixel_points[i]["phase"]], outline="black", width=max(1, scale//2))
            # Label placement only: never displace the model's dots.
            dx = 4 * scale if i % 2 == 0 else -11 * scale
            tx = min(max(0, x + dx), base.width - 12*scale)
            ty = min(max(0, y - 5*scale), base.height - 11*scale)
            draw.text((tx, ty), str(i), font=font(9*scale), fill=COLORS[pixel_points[i]["phase"]],
                      stroke_width=max(1, scale//2), stroke_fill="black")
        if scale == 4:
            panel = Image.new("RGB", (base.width + 360, base.height), "#111827")
            panel.paste(base, (0, 0))
            d = ImageDraw.Draw(panel)
            left = base.width + 20
            d.text((left, 24), "MODEL-PREDICTED TCP PATH", font=font(19), fill="white")
            d.text((left, 58), "Grasp gray cup, then lift", font=font(19), fill="white")
            d.text((left, 93), "Original points; no manual correction", font=font(14), fill="#cbd5e1")
            d.text((left, 118), "2D hypothesis, not robot commands", font=font(14), fill="#cbd5e1")
            for i, p in enumerate(pixel_points):
                text = f'{p["step"]:02d}  {p["phase"]:8s} ({p["pixel_x"]:.1f}, {p["pixel_y"]:.1f})'
                d.text((left, 170 + i*36), text, font=font(16), fill=COLORS[p["phase"]])
            base = panel
        base.save(output_dir / name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    render(args.image, json.loads(Path(args.trajectory).read_text()), args.output_dir)
