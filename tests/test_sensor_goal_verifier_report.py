"""Published-report metrics keep sensor errors and abstentions explicit."""
import unittest

from dvidia_training.sensor_goal_verifier_report import COHORTS, summarize


class SensorGoalReportTests(unittest.TestCase):
    def test_summary_keeps_false_completion_and_lost_completion_denominators(self):
        cases=[
            dict(cohort='native',physical_truth={'complete':True},generation_valid=True,
                 baseline={'verdict':'complete'},fused={'verdict':'unknown'},timing_ms={'processed_window':[2.]}),
            dict(cohort='primary_forgery',physical_truth={'complete':False},generation_valid=True,
                 baseline={'verdict':'complete'},fused={'verdict':'incomplete'},timing_ms={'processed_window':[3.]}),
            dict(cohort='cross_channel',physical_truth={'complete':False},generation_valid=False,
                 baseline={'verdict':'incomplete'},fused={'verdict':'incomplete'},timing_ms={'processed_window':[4.]})]
        rows=summarize(cases)
        self.assertEqual([row['cohort'] for row in rows],[*COHORTS,'all'])
        all_rows=rows[-1]
        self.assertEqual(all_rows['planned'],3)
        self.assertEqual(all_rows['generation_invalid'],1)
        self.assertEqual(all_rows['baseline']['false_completion'],{'numerator':1,'denominator':2})
        self.assertEqual(all_rows['fused']['false_completion'],{'numerator':0,'denominator':2})
        self.assertEqual(all_rows['paired']['caught_false_completions'],{'numerator':1,'denominator':1})
        self.assertEqual(all_rows['paired']['lost_true_completions'],{'numerator':1,'denominator':1})
        self.assertEqual(all_rows['latency_ms']['processed_window']['n'],3)


if __name__=='__main__': unittest.main()
