import copy
from itertools import permutations
import unittest
from unittest.mock import patch

from dvidia_training import tidy_table as t


class PlannerTests(unittest.TestCase):
    def test_all_assignments_and_true_noops(self):
        ref = t.remember_table(t.HOME)
        costs = []
        for perm in permutations(t.ZONES):
            scene = dict(zip(t.OBJECTS, perm))
            result = t.plan_table(ref, scene)
            for move in result["moves"]:
                self.assertEqual(scene[move["object"]], move["from"])
                self.assertNotIn(move["to"], scene.values())
                scene[move["object"]] = move["to"]
            self.assertEqual(scene, t.HOME)
            if perm[-1] == "buffer":
                costs.append(len(result["moves"]))
        self.assertEqual(sorted(costs), [0, 3, 3, 3, 4, 4])

    def test_equivalent_arrangement_and_uncertainty(self):
        ref = t.remember_table(t.HOME, [["A", "B"]])
        other = dict(A="center", B="left", C="right")
        self.assertEqual(t.plan_table(ref, other)["status"], "tidy")
        for value in [None, "center"]:
            ambiguous = dict(other, B=value)
            self.assertEqual(t.plan_table(ref, ambiguous)["status"], "unknown")
            with self.assertRaises(ValueError):
                t.remember_table(ambiguous)
        forged = copy.deepcopy(ref)
        forged["allowed_zones"]["C"].append("buffer")
        with self.assertRaises(ValueError):
            t.plan_table(forged, other)
        with self.assertRaises(ValueError):
            t.remember_table(dict(t.HOME, B="buffer"))

    def test_capture_uses_observation(self):
        scene = MockNative(dict(A="right", B="left", C="center"))
        with patch.object(t, "_native_factory", return_value=lambda *args, **kw: scene):
            saved = t.capture_table(t.HOME, assets=None)
        self.assertEqual(saved["reference"]["remembered"], scene.state)
        self.assertNotEqual(saved["reference"]["remembered"], t.HOME)


class MockNative:
    def __init__(self, state, *, frozen=False, valid=True, accept_move=True):
        self.state, self.frozen = dict(state), frozen
        self.valid, self.accept_move = valid, accept_move
        self.observations = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def observe(self):
        self.observations += 1
        return self.state.copy()

    def move(self, obj, to):
        if not self.frozen:
            self.state[obj] = to
        return dict(first_motion_ms=.1, verified=self.accept_move)

    def score(self, allowed):
        return dict(success=self.valid and all(
            zone in allowed[obj] for obj, zone in self.state.items()))


class RunnerTests(unittest.TestCase):
    def run_case(self, sim):
        with patch.object(t, "_native_factory", return_value=lambda *args, **kwargs: sim):
            return t.run_table_case(t.remember_table(t.HOME), t.HOME, assets=None)

    def test_reobserves_and_times_real_motion_separately_from_noop(self):
        sim = MockNative(dict(A="center", B="left", C="right"))
        receipt = self.run_case(sim)
        self.assertTrue(receipt["success"])
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(len(receipt["moves"]), 3)
        self.assertEqual(sim.observations, 4)
        self.assertGreater(receipt["time_to_first_move_ms"], receipt["first_decision_ms"])
        tidy = self.run_case(MockNative(t.HOME))
        self.assertEqual(tidy["status"], "tidy")
        self.assertEqual(tidy["moves"], [])
        self.assertIsNone(tidy["time_to_first_move_ms"])
        self.assertIsNotNone(tidy["first_decision_ms"])

    def test_false_completion_unknown_and_first_failure(self):
        failure = self.run_case(MockNative(t.HOME, valid=False))
        self.assertFalse(failure["success"])
        self.assertTrue(failure["camera_claimed_tidy"])
        unknown = self.run_case(MockNative(dict(t.HOME, C=None)))
        self.assertEqual(unknown["status"], "unknown")
        self.assertEqual(unknown["moves"], [])
        fails_move = self.run_case(MockNative(dict(t.HOME, B="buffer"), accept_move=False))
        self.assertFalse(fails_move["success"])
        self.assertEqual(len(fails_move["moves"]), 1)
        self.assertIsNotNone(fails_move["time_to_first_move_ms"])
        repeated = self.run_case(MockNative(dict(t.HOME, B="buffer"), frozen=True))
        self.assertFalse(repeated["success"])
        self.assertEqual(len(repeated["moves"]), 1)
