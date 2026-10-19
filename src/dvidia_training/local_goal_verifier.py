"""Small deterministic verifier for a fixed, sampled PiPER placement contract.

The goal and current clock reading are trusted caller inputs. Observation
packets cannot change thresholds, frames, goals, or verdicts. This checks a
declared 2 ms sensor stream; it does not authenticate the sensor, estimate
contact forces, establish continuous-time safety, or operate robot actuators.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType

SCHEMA = "dvidia.local-goal-observations/v1"
RESULT_SCHEMA = "dvidia.local-goal-verdict/v1"
SAMPLE_PERIOD_S = .002
REQUIRED_SAMPLES = 100
MAX_SAMPLES = 500
MAX_AGE_S = .100
TIME_TOLERANCE_S = .000001
MAX_CLOCK_S = 1_000_000_000.
UNITS = MappingProxyType({"position":"m", "angle":"rad", "time":"s",
                         "linear_speed":"m/s", "angular_speed":"rad/s", "force":"N"})
CRITERIA = MappingProxyType({"position_tolerance_m":.025,
    "yaw_tolerance_rad":math.radians(15), "linear_speed_max_m_s":.01,
    "angular_speed_max_rad_s":.05, "support_force_min_n":.01,
    "robot_force_max_n":.01, "continuous_dwell_s":.200})
LIMITATIONS = (
    "Self-consistent fabricated sensor observations cannot be detected by this verifier; sensor provenance and integrity need an independent trusted channel.",
    "A pass means the latest 100 samples satisfy the fixed contract at a declared 2 ms cadence; it is not continuous-time, hardware, identity-estimator or force-sensor certification.",
    "The caller must provide a trusted goal and now_s on the same monotonic or simulation clock as the observations; the packet cannot authenticate its own clock.",
    "Uncertainty bounds are asserted by the data source, not validated calibration. Explicit zero bounds represent an exact native-state assumption; they do not establish noiseless physical sensors.",
)
_PACKET_FIELDS = {"schema","units","uncertainty_bounds","sample_period_s","history_start_s","samples"}
_MEASUREMENTS = ("position_m","yaw_rad","linear_speed_m_s","angular_speed_rad_s",
                 "table_support_force_n","robot_contact_force_n")
_UNCERTAINTY_MAX = {"position_m":2.,"yaw_rad":math.pi,"linear_speed_m_s":20.,
                    "angular_speed_rad_s":100.,"table_support_force_n":1000.,"robot_contact_force_n":1000.}
_SAMPLE_FIELDS = {"t_s","object_id","frame_id","presence",*_MEASUREMENTS}
_PREDICATES = ("contract","schema","units","cadence","timestamps","freshness",
               "identity","frame","measurement_validity","uncertainty_validity","history_span","presence",
               "xy_position","yaw","linear_speed","angular_speed","support","release")


def _number(value, low, high):
    # Bound arbitrary-size JSON integers before math.isfinite converts to float.
    return type(value) in (int,float) and low<=value<=high and math.isfinite(value)


def _identifier(value):
    return (type(value) is str and 1<=len(value)<=96 and
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*",value) is not None)


@dataclass(frozen=True)
class GoalContract:
    """Trusted placement goal; thresholds are fixed by this module, not callers."""
    object_id: str
    frame_id: str
    goal_xy_m: tuple[float,float]
    goal_yaw_rad: float

    def __post_init__(self):
        if not _identifier(self.object_id) or not _identifier(self.frame_id):
            raise ValueError("Goal object/frame identifiers must be bounded explicit identifiers")
        if (type(self.goal_xy_m) not in (tuple,list) or len(self.goal_xy_m)!=2
                or not all(_number(v,-2.,2.) for v in self.goal_xy_m)):
            raise ValueError("Goal XY requires two finite coordinates in [-2,2] metres")
        if not _number(self.goal_yaw_rad,-math.pi,math.pi):
            raise ValueError("Goal yaw requires a finite angle in [-pi,pi] radians")
        object.__setattr__(self,"goal_xy_m",tuple(float(v) for v in self.goal_xy_m))
        object.__setattr__(self,"goal_yaw_rad",float(self.goal_yaw_rad))


def contract_description():
    """Portable public specification, independent of any observation packet."""
    return {"schema":SCHEMA,"units":dict(UNITS),"criteria":dict(CRITERIA),
            "sample_period_s":SAMPLE_PERIOD_S,"required_samples":REQUIRED_SAMPLES,
            "max_samples":MAX_SAMPLES,"max_age_s":MAX_AGE_S,
            "timestamp_tolerance_s":TIME_TOLERANCE_S,
            "bounds":{"position_m":[-2.,2.],"yaw_rad":[-math.pi,math.pi],
                      "linear_speed_m_s":[0.,20.],"angular_speed_rad_s":[0.,100.],
                      "force_n":[0.,1000.],"clock_s":[0.,MAX_CLOCK_S]},
            "packet_fields":sorted(_PACKET_FIELDS),"sample_fields":sorted(_SAMPLE_FIELDS),
            "uncertainty_maximum_bounds":dict(_UNCERTAINTY_MAX),
            "uncertainty_semantics":"Mandatory symmetric absolute bounds in declared SI units. position_m bounds XY error norm, yaw_rad bounds wrapped yaw-distance error. Nonnegative metrics have lower bounds clamped at zero; yaw-distance upper is clamped at pi. All selected samples share these asserted bounds.",
            "history_semantics":"history_start_s is the beginning of the first sample interval. Each t_s ends one 2 ms interval. Validate the full packet; score only the latest 100 samples.",
            "absence_semantics":"Absent is a positive observation that the named target is absent, not omitted sensor data. Absent/unknown observations require null pose, speed and force fields.",
            "decision_semantics":"Invalid, missing, stale, gapped or insufficient evidence returns unknown before physical scoring. Unknown presence in the scoring window returns unknown. A physical predicate fails only when its entire interval violates its threshold. Any definite failure returns incomplete; otherwise any threshold overlap returns unknown; all intervals passing returns complete.",
            "limitations":list(LIMITATIONS)}


class _InvalidEvidence(Exception):
    def __init__(self,predicate,reason,detail):
        self.predicate,self.reason,self.detail=predicate,reason,detail


def verify_goal(packet, *, contract, now_s):
    """Return a tri-state verdict and predicate reasons; never read truth labels.

    Invalid JSON-like observation data returns ``unknown``. Malformed trusted
    goals are normally rejected by GoalContract construction; a wrong contract
    type here also returns unknown. No network, native runtime or file access.
    """
    predicates={name:{"status":"unknown","reason":"not_evaluated"} for name in _PREDICATES}
    window={}

    def check(condition,predicate,reason,detail):
        if not condition:
            raise _InvalidEvidence(predicate,reason,detail)

    def passed(predicate,reason="validated",**detail):
        predicates[predicate]={"status":"pass","reason":reason,**detail}

    def result(verdict):
        reasons=[value["reason"] for value in predicates.values()
                 if value["status"] in ("fail","unknown") and value["reason"]!="not_evaluated"]
        return {"schema":RESULT_SCHEMA,"verdict":verdict,"reasons":reasons,
                "predicates":predicates,"window":window,"criteria":dict(CRITERIA),
                "limitations":list(LIMITATIONS)}

    try:
        check(type(contract) is GoalContract,"contract","invalid_goal_contract",
              "Use a trusted GoalContract rather than packet-provided goals or thresholds.")
        # Revalidate even a dataclass altered through low-level Python machinery.
        check(_identifier(contract.object_id) and _identifier(contract.frame_id)
              and type(contract.goal_xy_m) is tuple and len(contract.goal_xy_m)==2
              and all(_number(v,-2.,2.) for v in contract.goal_xy_m)
              and _number(contract.goal_yaw_rad,-math.pi,math.pi),
              "contract","invalid_goal_contract","Trusted goal has invalid bounds or identifiers.")
        passed("contract")
        check(_number(now_s,0.,MAX_CLOCK_S),"freshness","invalid_trusted_clock",
              "now_s must be a finite nonnegative clock value in seconds.")
        check(type(packet) is dict and set(packet)==_PACKET_FIELDS,
              "schema","invalid_packet_fields","Packet has missing or unexpected fields.")
        check(type(packet["schema"]) is str and packet["schema"]==SCHEMA,
              "schema","unsupported_schema","Unsupported observation schema.")
        passed("schema")
        check(type(packet["units"]) is dict and packet["units"]==dict(UNITS),
              "units","invalid_units","Require the exact declared SI unit map.")
        passed("units")
        uncertainty=packet["uncertainty_bounds"]
        check(type(uncertainty) is dict and set(uncertainty)==set(_UNCERTAINTY_MAX),
              "uncertainty_validity","invalid_uncertainty_fields","Require all six explicit uncertainty bounds.")
        for key,upper in _UNCERTAINTY_MAX.items():
            check(_number(uncertainty[key],0.,upper),"uncertainty_validity","invalid_uncertainty_bound",
                  f"{key} uncertainty must be a finite nonnegative bound no larger than {upper}.")
        passed("uncertainty_validity",reason="source_asserted_bounds_valid")
        check(_number(packet["sample_period_s"],SAMPLE_PERIOD_S,SAMPLE_PERIOD_S),
              "cadence","invalid_sample_period","This contract accepts only declared 2 ms samples.")
        passed("cadence")
        check(_number(packet["history_start_s"],0.,MAX_CLOCK_S),
              "timestamps","invalid_history_start","Invalid history interval start.")
        samples=packet["samples"]
        check(type(samples) is list and len(samples)<=MAX_SAMPLES,
              "history_span","invalid_sample_count","Require a list of at most 500 observations.")
        check(bool(samples),"history_span","insufficient_history","No observations are available.")
        previous=float(packet["history_start_s"])
        for index,sample in enumerate(samples):
            check(type(sample) is dict and set(sample)==_SAMPLE_FIELDS,
                  "schema","invalid_sample_fields",f"Sample {index} has missing or unexpected fields.")
            t=sample["t_s"]
            check(_number(t,0.,MAX_CLOCK_S),"timestamps","invalid_timestamp",f"Invalid timestamp at sample {index}.")
            check(t>previous,"timestamps","nonmonotonic_timestamps",f"Timestamp did not advance at sample {index}.")
            check(abs((t-previous)-SAMPLE_PERIOD_S)<=TIME_TOLERANCE_S,
                  "timestamps","gapped_or_irregular_history",f"Sample {index} does not end the next 2 ms interval.")
            check(abs(t-(packet["history_start_s"]+(index+1)*SAMPLE_PERIOD_S))<=TIME_TOLERANCE_S,
                  "timestamps","gapped_or_irregular_history",f"Sample {index} drifts from the declared interval clock.")
            previous=float(t)
            check(_identifier(sample["object_id"]) and sample["object_id"]==contract.object_id,
                  "identity","unverified_target_identity",f"Sample {index} does not identify the trusted target.")
            check(_identifier(sample["frame_id"]) and sample["frame_id"]==contract.frame_id,
                  "frame","invalid_reference_frame",f"Sample {index} does not use the trusted goal frame.")
            check(type(sample["presence"]) is str and sample["presence"] in ("present","absent","unknown"),
                  "measurement_validity","invalid_presence",f"Invalid presence evidence at sample {index}.")
            if sample["presence"]!="present":
                check(all(sample[key] is None for key in _MEASUREMENTS),
                      "measurement_validity","contradictory_presence_measurements",
                      f"Sample {index} reports absent/unknown presence with numeric object measurements.")
                continue
            p=sample["position_m"]
            check(type(p) is list and len(p)==3 and all(_number(v,-2.,2.) for v in p),
                  "measurement_validity","invalid_position",f"Sample {index} needs XYZ in metres, each within [-2,2].")
            bounds={"yaw_rad":(-math.pi,math.pi),"linear_speed_m_s":(0.,20.),
                    "angular_speed_rad_s":(0.,100.),"table_support_force_n":(0.,1000.),
                    "robot_contact_force_n":(0.,1000.)}
            for key,(low,high) in bounds.items():
                check(_number(sample[key],low,high),"measurement_validity","invalid_"+key,
                      f"Sample {index} has invalid or out-of-range {key}.")
        passed("timestamps")
        passed("identity")
        passed("frame")
        passed("measurement_validity")
        newest=float(samples[-1]["t_s"])
        age=float(now_s)-newest
        window.update(source_sample_count=len(samples),newest_t_s=newest,age_s=age,
                      declared_sample_period_s=SAMPLE_PERIOD_S)
        check(age>=-TIME_TOLERANCE_S,"freshness","future_observation","Newest observation is after the trusted clock.")
        check(age<=MAX_AGE_S+TIME_TOLERANCE_S,"freshness","stale_observation","Newest observation is more than 100 ms old.")
        passed("freshness",age_s=age,max_age_s=MAX_AGE_S)
        check(len(samples)>=REQUIRED_SAMPLES,"history_span","insufficient_history",
              "At least 100 contiguous 2 ms samples are required before physical scoring.")
        selected=samples[-REQUIRED_SAMPLES:]
        start=(float(packet["history_start_s"]) if len(samples)==REQUIRED_SAMPLES
               else float(samples[-REQUIRED_SAMPLES-1]["t_s"]))
        duration=newest-start
        check(abs(duration-REQUIRED_SAMPLES*SAMPLE_PERIOD_S)<=TIME_TOLERANCE_S,
              "timestamps","invalid_window_duration","Latest 100 intervals do not span 200 ms within the single clock tolerance.")
        window.update(evaluated_sample_count=REQUIRED_SAMPLES,
                      evaluated_start_s=start,evaluated_end_s=newest,evaluated_duration_s=duration,
                      evaluated_declared_duration_s=REQUIRED_SAMPLES*SAMPLE_PERIOD_S)
        passed("history_span",reason="latest_200ms_available")
        check(not any(row["presence"]=="unknown" for row in selected),
              "presence","unknown_target_presence","Target presence is unknown within the latest 200 ms.")
        present=[row for row in selected if row["presence"]=="present"]
        if len(present)!=REQUIRED_SAMPLES:
            predicates["presence"]={"status":"fail","reason":"target_absent_in_window",
                                     "absent_samples":REQUIRED_SAMPLES-len(present)}
            for key in ("xy_position","yaw","linear_speed","angular_speed","support","release"):
                predicates[key]={"status":"unknown","reason":"not_applicable_target_absent"}
            return result("incomplete")
        passed("presence",reason="target_present_in_every_sample")
        xy=[math.hypot(row["position_m"][0]-contract.goal_xy_m[0],
                       row["position_m"][1]-contract.goal_xy_m[1]) for row in selected]
        yaw=[abs(math.atan2(math.sin(row["yaw_rad"]-contract.goal_yaw_rad),
                            math.cos(row["yaw_rad"]-contract.goal_yaw_rad))) for row in selected]
        checks=(
            ("xy_position",max(xy),"position_m",CRITERIA["position_tolerance_m"],"max","le","position_outside_goal"),
            ("yaw",max(yaw),"yaw_rad",CRITERIA["yaw_tolerance_rad"],"max","le","yaw_outside_goal"),
            ("linear_speed",max(row["linear_speed_m_s"] for row in selected),"linear_speed_m_s",CRITERIA["linear_speed_max_m_s"],"max","le","linear_motion_in_window"),
            ("angular_speed",max(row["angular_speed_rad_s"] for row in selected),"angular_speed_rad_s",CRITERIA["angular_speed_max_rad_s"],"max","le","angular_motion_in_window"),
            ("support",min(row["table_support_force_n"] for row in selected),"table_support_force_n",CRITERIA["support_force_min_n"],"min","gt","insufficient_table_support"),
            ("release",max(row["robot_contact_force_n"] for row in selected),"robot_contact_force_n",CRITERIA["robot_force_max_n"],"max","le","robot_contact_in_window"))
        failures=uncertain=0
        for predicate,observed,bound_key,threshold,statistic,operator,reason in checks:
            radius=uncertainty[bound_key]
            lower=max(0.,observed-radius)
            upper=min(math.pi,observed+radius) if predicate=="yaw" else observed+radius
            okay=upper<=threshold if operator=="le" else lower>threshold
            failed=lower>threshold if operator=="le" else upper<=threshold
            status="pass" if okay else "fail" if failed else "unknown"
            failures+=failed
            uncertain+=not okay and not failed
            predicates[predicate]={"status":status,
                "reason":"interval_within_threshold_for_every_sample" if okay else reason if failed else "uncertainty_overlaps_threshold",
                "observed":observed,"interval":{"lower":lower,"upper":upper},
                "asserted_uncertainty_bound":radius,"statistic":statistic,"threshold":threshold,"operator":operator}
        return result("incomplete" if failures else "unknown" if uncertain else "complete")
    except _InvalidEvidence as issue:
        predicates[issue.predicate]={"status":"unknown","reason":issue.reason,"detail":issue.detail}
        return result("unknown")
