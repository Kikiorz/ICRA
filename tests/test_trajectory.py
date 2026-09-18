import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from trajectory import parse_response, to_pixel, validate
from render import render


def fixture():
    phases = ["start", "approach", "approach", "approach", "grasp", "lift", "lift", "lift"]
    return {"task": "synthetic test", "coordinate_system": "normalized_1000", "tracked_point": "gripper_tcp",
            "image_size": [256, 128], "target": {"description": "test", "bbox": [100, 100, 400, 500]},
            "grasp_strategy": "test only", "assumptions": [],
            "waypoints": [{"step": i, "x": 200+i*30, "y": 300,
                           "phase": phase, "gripper": {"start": "open", "approach": "open", "grasp": "close", "lift": "closed"}[phase]}
                          for i, phase in enumerate(phases)]}


class TrajectoryTests(unittest.TestCase):
    def test_normalization_uses_original_non_square_image(self):
        self.assertEqual(to_pixel(1000, 1000, (256, 128)), (255, 127))
        self.assertEqual(to_pixel(0, 0, (256, 128)), (0, 0))

    def test_reject_invalid_coordinates_without_repair(self):
        for value in (1001, -1, float("nan"), True, "500"):
            data = fixture()
            data["waypoints"][3]["x"] = value
            with self.assertRaises(ValueError):
                validate(data, (256, 128))

    def test_reject_lift_before_grasp(self):
        data = fixture()
        data["waypoints"][3].update(phase="lift", gripper="closed")
        with self.assertRaises(ValueError):
            validate(data, (256, 128))

    def test_parse_final_answer_only(self):
        data = fixture()
        self.assertEqual(parse_response('analysis {bad}</think>\n```json\n'+json.dumps(data)+'\n```'), data)
        with self.assertRaises(ValueError):
            parse_response('{"incomplete":')

    def test_renderer_preserves_source_and_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.png"
            Image.new("RGB", (256, 128), (20, 30, 40)).save(source)
            original = source.read_bytes()
            data = fixture()
            before = copy.deepcopy(data)
            render(source, data, root / "out")
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(data, before)
            overlay = Image.open(root / "out/overlay.png")
            self.assertEqual(overlay.size, (256, 128))
            self.assertEqual(overlay.getpixel((255, 127)), (20, 30, 40))
            points = json.loads((root / "out/pixel_waypoints.json").read_text())
            self.assertEqual(points[0]["x"], data["waypoints"][0]["x"])
            self.assertEqual(points[0]["pixel_x"], 51)


if __name__ == "__main__":
    unittest.main()
