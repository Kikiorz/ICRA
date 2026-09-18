import unittest

from hamster.trajectory import parse_response


class HamsterParserTests(unittest.TestCase):
    def test_preserves_points_and_ordered_gripper_events(self):
        result = parse_response('<ans>[(0.2, 0.3), <action>Open Gripper</action>, (0.4, 0.6), <action>Close Gripper</action>, (0.4, 0.3)]</ans>')
        self.assertEqual([(p['x'], p['y']) for p in result['waypoints']], [(0.2, 0.3), (0.4, 0.6), (0.4, 0.3)])
        self.assertEqual(result['gripper_events'], [{'after_step': 0, 'action': 'Open Gripper'}, {'after_step': 1, 'action': 'Close Gripper'}])

    def test_rejects_code_invalid_coordinates_and_malformed_answers(self):
        for text in ["<ans>[__import__('os').system('false')]</ans>",
                     '<ans>[(True, 0.3), (0.2, 0.4)]</ans>',
                     '<ans>[(1.1, 0.3), (0.2, 0.4)]</ans>',
                     '<ans>[<action>Open Gripper</action>, (0.2, 0.3)]</ans>',
                     '<ans>[(0.2, 0.3)]',
                     '<ans>[(0.2, 0.3)]</ans><ans>[(0.2, 0.3)]</ans>']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response(text)


if __name__ == '__main__':
    unittest.main()
