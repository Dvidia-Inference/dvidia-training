"""Adversarial evidence controls for the local physical-goal verifier."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import math
import unittest

from dvidia_training import local_goal_verifier as verifier


def packet(count=100, start=.4):
    return {"schema":verifier.SCHEMA,"units":dict(verifier.UNITS),"sample_period_s":.002,
            "uncertainty_bounds":{key:0. for key in ("position_m","yaw_rad","linear_speed_m_s",
                "angular_speed_rad_s","table_support_force_n","robot_contact_force_n")},
            "history_start_s":start,"samples":[{
                "t_s":start+(i+1)*.002,"object_id":"green-shoe","frame_id":"workstation",
                "presence":"present","position_m":[.4,-.1,0.],"yaw_rad":0.,
                "linear_speed_m_s":0.,"angular_speed_rad_s":0.,
                "table_support_force_n":.2943,"robot_contact_force_n":0.}
                for i in range(count)]}


class LocalGoalVerifierTests(unittest.TestCase):
    def setUp(self):
        self.goal=verifier.GoalContract("green-shoe","workstation",(.4,-.1),0.)

    def verify(self,p,now=None):
        if now is None:
            now=p["samples"][-1]["t_s"] if p["samples"] else .6
        return verifier.verify_goal(p,contract=self.goal,now_s=now)

    def test_exact_complete_window_and_trusted_immutable_goal(self):
        result=self.verify(packet())
        self.assertEqual(result["verdict"],"complete")
        self.assertAlmostEqual(result["window"]["evaluated_duration_s"],.2)
        self.assertEqual(result["reasons"],[])
        self.assertTrue(all(p["status"]=="pass" for p in result["predicates"].values()))
        with self.assertRaises(FrozenInstanceError):self.goal.goal_yaw_rad=1.
        with self.assertRaises(ValueError):verifier.GoalContract("shoe","world",(True,0),0)
        with self.assertRaises(ValueError):verifier.GoalContract("shoe","world",(0,0),float("nan"))

    def test_each_physical_failure_is_incomplete_with_its_reason(self):
        changes=(
            ("position_m",[.5,-.1,0.],"xy_position"),
            ("yaw_rad",.5,"yaw"),
            ("linear_speed_m_s",.12,"linear_speed"),
            ("angular_speed_rad_s",.1,"angular_speed"),
            ("table_support_force_n",.01,"support"),
            ("robot_contact_force_n",.02,"release"))
        for key,value,predicate in changes:
            with self.subTest(key=key):
                p=packet();p["samples"][-25][key]=value
                result=self.verify(p)
                self.assertEqual(result["verdict"],"incomplete")
                self.assertEqual(result["predicates"][predicate]["status"],"fail")

    def test_only_latest_window_controls_dwell(self):
        p=packet(150)
        p["samples"][49]["linear_speed_m_s"]=.12
        self.assertEqual(self.verify(p)["verdict"],"complete")
        p["samples"][50]["linear_speed_m_s"]=.12
        self.assertEqual(self.verify(p)["verdict"],"incomplete")

    def test_short_or_empty_history_unknown_before_physical_scoring(self):
        for count in (0,1,99):
            with self.subTest(count=count):
                p=packet(count)
                if count:p["samples"][-1]["position_m"]=[.7,0,0]
                result=self.verify(p)
                self.assertEqual(result["verdict"],"unknown")
                self.assertIn("insufficient_history",result["reasons"])
        self.assertEqual(self.verify(packet(501))["verdict"],"unknown")

    def test_gapped_duplicate_reordered_and_claimed_cadence_fail_closed(self):
        mutations=(lambda p:p["samples"][50].update(t_s=p["samples"][50]["t_s"]+.002),
            lambda p:p["samples"][50].update(t_s=p["samples"][49]["t_s"]),
            lambda p:p["samples"].reverse(),lambda p:p.update(sample_period_s=.2),
            lambda p:p.update(history_start_s=.399))
        for mutate in mutations:
            p=packet();mutate(p)
            self.assertEqual(self.verify(p,now=.6)["verdict"],"unknown")
        p=packet()
        for index,row in enumerate(p["samples"]):row["t_s"]=.4+(index+1)*.0019995
        self.assertEqual(self.verify(p,now=.6)["verdict"],"unknown")

    def test_freshness_uses_trusted_now_and_rejects_future_or_invalid_clock(self):
        self.assertEqual(self.verify(packet(),now=.7)["verdict"],"complete")
        for now in (.70001,.599,float("nan"),float("inf"),True,-1.,10**1000):
            with self.subTest(now=now):
                self.assertEqual(self.verify(packet(),now=now)["verdict"],"unknown")

    def test_frame_identity_units_and_goal_spoofs_are_unknown(self):
        mutations=(lambda p:p["samples"][0].update(frame_id="camera"),
            lambda p:p["samples"][0].update(object_id="orange-shoe"),
            lambda p:p["units"].update(position="cm"),
            lambda p:p.update(goal_xy_m=[.7,0]),lambda p:p.update(complete=True),
            lambda p:p.update(criteria={"position_tolerance_m":99.}),
            lambda p:p["samples"][0].update(instantaneous_conditions_met=True))
        for mutate in mutations:
            p=packet();mutate(p)
            self.assertEqual(self.verify(p)["verdict"],"unknown")
        self.assertEqual(verifier.verify_goal(packet(),contract={},now_s=.6)["verdict"],"unknown")

    def test_invalid_numeric_values_cannot_be_mistaken_for_measurements(self):
        for key in ("yaw_rad","linear_speed_m_s","angular_speed_rad_s",
                    "table_support_force_n","robot_contact_force_n"):
            for value in (None,True,"0",float("nan"),float("inf"),-1001.,1001.,10**1000):
                with self.subTest(key=key,value=value):
                    p=packet();p["samples"][3][key]=value
                    self.assertEqual(self.verify(p)["verdict"],"unknown")
        for value in ([True,0.,0.],[3.,0.,0.],[0.,0.],"0,0,0"):
            p=packet();p["samples"][3]["position_m"]=value
            self.assertEqual(self.verify(p)["verdict"],"unknown")
        for key in ("sample_period_s","history_start_s"):
            p=packet();p[key]=10**1000
            self.assertEqual(self.verify(p)["verdict"],"unknown")

    def test_absence_is_incomplete_unknown_presence_and_contradiction_unknown(self):
        for presence,expected in (("absent","incomplete"),("unknown","unknown")):
            p=packet();row=p["samples"][-1];row["presence"]=presence
            for key in ("position_m","yaw_rad","linear_speed_m_s","angular_speed_rad_s",
                        "table_support_force_n","robot_contact_force_n"):row[key]=None
            self.assertEqual(self.verify(p)["verdict"],expected)
        p=packet();p["samples"][-1]["presence"]="absent"
        self.assertEqual(self.verify(p)["verdict"],"unknown")

    def test_validation_first_even_when_a_physical_failure_exists(self):
        p=packet();p["samples"][-1]["robot_contact_force_n"]=.8
        p["samples"][0]["yaw_rad"]=float("nan")
        self.assertEqual(self.verify(p)["verdict"],"unknown")
        p=packet(150);p["samples"][0]["frame_id"]="untrusted"
        self.assertEqual(self.verify(p)["verdict"],"unknown")

    def test_yaw_wrap_and_threshold_edges(self):
        goal=verifier.GoalContract("green-shoe","workstation",(.4,-.1),math.pi-.01)
        p=packet()
        for row in p["samples"]:
            row["yaw_rad"]=-math.pi+.01
            row["linear_speed_m_s"]=.01
            row["angular_speed_rad_s"]=.05
            row["robot_contact_force_n"]=.01
        result=verifier.verify_goal(p,contract=goal,now_s=.6)
        self.assertEqual(result["verdict"],"complete")
        self.assertAlmostEqual(result["predicates"]["yaw"]["observed"],.02)

    def test_inputs_are_not_mutated_and_fabrication_limitation_is_explicit(self):
        p=packet();before=deepcopy(p)
        result=self.verify(p)
        self.assertEqual(p,before)
        self.assertEqual(result["verdict"],"complete")
        self.assertIn("fabricated",result["limitations"][0])

    def test_missing_invalid_or_partial_uncertainty_is_unknown(self):
        p=packet();del p["uncertainty_bounds"]
        self.assertEqual(self.verify(p)["verdict"],"unknown")
        p=packet();del p["uncertainty_bounds"]["yaw_rad"]
        self.assertEqual(self.verify(p)["verdict"],"unknown")
        for value in (-.001,True,"0",float("nan"),float("inf"),1001.):
            p=packet();p["uncertainty_bounds"]["position_m"]=value
            self.assertEqual(self.verify(p)["verdict"],"unknown")

    def test_uncertainty_crossing_threshold_is_unknown_and_definite_failure_wins(self):
        p=packet()
        for row in p["samples"]:row["linear_speed_m_s"]=.009
        p["uncertainty_bounds"]["linear_speed_m_s"]=.002
        result=self.verify(p)
        self.assertEqual(result["verdict"],"unknown")
        self.assertEqual(result["predicates"]["linear_speed"]["status"],"unknown")
        p["samples"][-1]["robot_contact_force_n"]=.8
        self.assertEqual(self.verify(p)["verdict"],"incomplete")

    def test_strict_support_interval_boundary_and_closed_release_boundary(self):
        p=packet()
        for row in p["samples"]:row["table_support_force_n"]=.02
        p["uncertainty_bounds"]["table_support_force_n"]=.01
        self.assertEqual(self.verify(p)["verdict"],"unknown")
        for row in p["samples"]:row["table_support_force_n"]=.005
        p["uncertainty_bounds"]["table_support_force_n"]=.005
        self.assertEqual(self.verify(p)["verdict"],"incomplete")
        p=packet()
        for row in p["samples"]:row["robot_contact_force_n"]=.005
        p["uncertainty_bounds"]["robot_contact_force_n"]=.005
        self.assertEqual(self.verify(p)["verdict"],"complete")
        p["uncertainty_bounds"]["robot_contact_force_n"]=.006
        self.assertEqual(self.verify(p)["verdict"],"unknown")

    def test_position_uncertainty_is_a_norm_bound_and_yaw_is_wrapped(self):
        p=packet();p["uncertainty_bounds"]["position_m"]=.026
        self.assertEqual(self.verify(p)["verdict"],"unknown")
        goal=verifier.GoalContract("green-shoe","workstation",(.4,-.1),math.pi-.01)
        p=packet();p["uncertainty_bounds"]["yaw_rad"]=.25
        for row in p["samples"]:row["yaw_rad"]=-math.pi+.01
        result=verifier.verify_goal(p,contract=goal,now_s=.6)
        self.assertEqual(result["verdict"],"unknown")
        self.assertAlmostEqual(result["predicates"]["yaw"]["interval"]["upper"],.27)


if __name__=="__main__":unittest.main()
