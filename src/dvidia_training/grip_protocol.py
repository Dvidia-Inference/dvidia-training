"""Sealed cohorts and independent continuous-grip scoring; no controller imports.

A recorded condition is evaluated on every native tick. These synthetic criteria
are not a calibrated damage, safety, tactile sensing or hardware guarantee.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

SCHEMA = 'dvidia.continuous-shoe-grip/v2'
FIXTURE = Path(__file__).with_name('grip_protocol_v02.json')
FIXTURE_SHA256 = '41ab821fa79596d57228b93c2b42c17f6df4b4bc42e0f25818cb472cb221e6e7'
_REQUIRED = ('t_s','tool_shoe_force_n','shoe_robot_force_n','left_finger_force_n',
             'right_finger_force_n','shoe_x_m','shoe_y_m','shoe_z_m','world_shoe_force_n')
_BOUNDS = {'shoe_x':(.69,.75),'shoe_y':(-.15,-.05),'mass':(.1,1.),
           'friction':(.1,1.5),'tool_length':(.1,.4),'lateral_error':(-.12,.12)}


def _fixture():
    raw=FIXTURE.read_bytes()
    if sha256(raw).hexdigest()!=FIXTURE_SHA256:
        raise ValueError('Sealed continuous-grip protocol changed; declare a new protocol revision')
    value=json.loads(raw)
    ids=set()
    for case in value['cases']:
        if case['id'] in ids:raise ValueError('Duplicate cohort ID')
        ids.add(case['id'])
        for name,(low,high) in _BOUNDS.items():
            if type(case[name]) not in (int,float) or not math.isfinite(case[name]) or not low<=case[name]<=high:
                raise ValueError('Invalid sealed case parameter')
    return value


def protocol(split='heldout'):
    """Build paired attempts; reading cases does not execute native simulations."""
    if split not in ('development','regression','heldout'):
        raise ValueError('split must be development, regression or heldout')
    value=_fixture()
    if split=='heldout':cases=value['cases']
    else:
        # Prior public data are allowed for development, never called unseen.
        from . import tool_reach
        cases=tool_reach.cases('development')
        if split=='regression':cases+=tool_reach.cases('heldout')
        cases=[{**row,'previous_split':row['split'],'split':split} for row in cases]
    attempts=[{'id':case['id']+'-'+policy,'task':'tool-reach','policy':policy,'case':deepcopy(case)}
              for case in cases for policy in ('baseline','recovery')]
    return {'schema':SCHEMA,'split':split,'cohort':value['cohort'] if split=='heldout' else 'published-v01-'+split,
            'fixture_sha256':FIXTURE_SHA256,'criteria':deepcopy(value['criteria']),
            'attempts':attempts,'comparison':value['comparison'],'design':value['design'],
            'previous_data':value['previous_data'],'cohort_access':value['cohort_access'],
            'scope':'Authored controller comparison, privileged native observations and synthetic objects; no hardware safety or transfer claim.'}


def _number(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Grip evidence requires finite numeric values')
    return float(value)


def _trace(result):
    columns=result.get('native_trace_columns');raw=result.get('native_trace')
    if (not isinstance(columns,list) or not all(isinstance(name,str) for name in columns) or len(columns)!=len(set(columns)) or not set(_REQUIRED)<=set(columns)
            or not isinstance(raw,list) or not 250<=len(raw)<=100000):
        raise ValueError('Missing bounded dense grip trace or required columns')
    indices={name:columns.index(name) for name in _REQUIRED}
    rows=[]
    for sample in raw:
        if not isinstance(sample,(list,tuple)) or len(sample)!=len(columns):raise ValueError('Malformed native trace row')
        values=[_number(value) for value in sample]
        row={key:values[index] for key,index in indices.items()}
        if any(row[key]<0 for key in _REQUIRED if key.endswith('_force_n')):raise ValueError('Negative contact normal force')
        other=row['shoe_robot_force_n']-row['tool_shoe_force_n']-row['left_finger_force_n']-row['right_finger_force_n']
        if other < -1e-6:raise ValueError('Inconsistent shoe contact-force decomposition')
        row['other_robot_force_n']=max(0.,other);rows.append(row)
    dt=.002;times=[r['t_s'] for r in rows]
    if abs(times[0])>1e-8 and abs(times[0]-dt)>1e-8:raise ValueError('Trace omits its initial native steps')
    if any(abs(b-a-dt)>1e-8 for a,b in zip(times,times[1:])):raise ValueError('Native trace has a missing, duplicated or nonuniform timestep')
    metrics=result.get('metrics',{})
    end=_number(metrics.get('simulation_seconds',metrics.get('simulated_seconds')))
    if abs(times[-1]-end)>1e-6:raise ValueError('Dense trace does not reach the declared episode end')
    steps=metrics.get('native_steps')
    if type(steps) is not int or steps!=round(end/dt):raise ValueError('Native step count disagrees with trace duration')
    if len(rows)!=steps+int(abs(times[0])<=1e-8):raise ValueError('Native trace coverage is incomplete')
    if type(result.get('valid')) is not bool:raise ValueError('Native validity must be explicit')
    return rows


def _predicate(row,strict=True):
    ok=(row['shoe_z_m']>.035 and row['left_finger_force_n']>.01 and row['right_finger_force_n']>.01
        and row['tool_shoe_force_n']<.01 and row['world_shoe_force_n']<.01)
    return bool(ok and (not strict or row['other_robot_force_n']<.01))


def _losses(rows):
    """Condition loss after 100 ms acquisition; recovery needs 100 ms requalification.

    Sustained loss is at least 20 ms of consecutive failed scoring predicates.
    Shorter condition gaps remain reported, without being labelled missed alarms.
    This audit is independent of the controller's decision predicate.
    """
    episodes=[];acquired=False;good=0;bad=0;active=None
    for row in rows:
        ok=_predicate(row);t=row['t_s']
        if ok:
            good+=1;bad=0
            if good>=50:
                acquired=True
                if active is not None:
                    active['requalified_s']=t;episodes.append(active);active=None
        else:
            good=0;bad+=1
            if acquired and active is None:
                active={'onset_s':t,'requalified_s':None,'max_failed_ticks':0}
            if active is not None:active['max_failed_ticks']=max(active['max_failed_ticks'],bad)
    if active is not None:episodes.append(active)
    for item in episodes:item['sustained']=item['max_failed_ticks']>=10
    return episodes


def _event_scores(result,rows):
    events=result.get('events',[])
    if not isinstance(events,list):raise ValueError('Events must be a list')
    for event in events:
        if not isinstance(event,dict) or not isinstance(event.get('type'),str):raise ValueError('Malformed grip event')
        if not 0<=_number(event.get('t'))<=rows[-1]['t_s']+1e-6:raise ValueError('Event lies outside recorded episode')
    episodes=_losses(rows);used=set();response_used=set();issued_used=set()
    detections=[e for e in events if e['type']=='grip_loss_detected']
    responses=[e for e in events if e['type']=='grip_recovery_started']
    issued=[e for e in events if e['type']=='recovery_control_issued']
    issued_by_request={}
    for index,event in enumerate(issued):
        request=_number(event.get('requested_at_s'))
        if request>event['t']+1e-8 or abs(event['t']/.002-round(event['t']/.002))>1e-6:
            raise ValueError('Issued recovery packet precedes request or lies off native clock')
        if not any(abs(e['t']-request)<1e-8 for e in responses):
            raise ValueError('Issued recovery packet has no matching recovery request')
        if request in issued_by_request:raise ValueError('Duplicate first recovery packet for one request')
        for key in ('issued_ctrl','requested_joint_target'):
            vector=event.get(key)
            if not isinstance(vector,list) or len(vector)!=7:raise ValueError('Malformed recovery control packet')
            for value in vector:_number(value)
        issued_by_request[request]=index
    for loss in episodes:
        end=loss['requalified_s'] if loss['requalified_s'] is not None else rows[-1]['t_s']
        match=next((i for i,e in enumerate(detections) if i not in used and loss['onset_s']-1e-8<=e['t']<=end+.02),None)
        loss.update(detected_s=None,recovery_request_s=None,issued_command_s=None,
                    detection_latency_s=None,request_latency_s=None,command_issue_latency_s=None)
        if match is None:continue
        used.add(match);event=detections[match];loss['detected_s']=event['t']
        loss['detection_latency_s']=max(0.,event['t']-loss['onset_s'])
        response=next((i for i,e in enumerate(responses) if i not in response_used and event['t']-1e-8<=e['t']<=end+.02),None)
        if response is not None:
            response_used.add(response);loss['recovery_request_s']=responses[response]['t']
            loss['request_latency_s']=max(0.,loss['recovery_request_s']-loss['onset_s'])
            packet=issued_by_request.get(loss['recovery_request_s'])
            if packet is not None:
                issued_used.add(packet);loss['issued_command_s']=issued[packet]['t']
                loss['command_issue_latency_s']=max(0.,loss['issued_command_s']-loss['onset_s'])
    return {'loss_episodes':episodes,'sustained_loss_count':sum(e['sustained'] for e in episodes),
            'sustained_losses_without_detection':sum(e['sustained'] and e['detected_s'] is None for e in episodes),
            'detection_count':len(detections),'recovery_request_count':len(responses),'recovery_command_count':len(issued),
            'requalifications_after_command':sum(e['issued_command_s'] is not None and e['requalified_s'] is not None and e['issued_command_s']<=e['requalified_s'] for e in episodes),
            'unmatched_detection_count':len(detections)-len(used),'unmatched_recovery_request_count':len(responses)-len(response_used),
            'unmatched_recovery_command_count':len(issued)-len(issued_used),
            'detection_scope':'Independent native scoring-condition loss after 100 ms acquisition. Recovery request and first issued native control packet are separate; packet issuance does not establish changed target, physical stopping or successful recovery. Planned releases can also break this condition, so uncovered episodes are not certified missed alarms. Unmatched events may use another conservative predicate; they are not automatically false alarms. Times exclude sensor/transport latency and hardware stopping time.'}


def score_result(result):
    """Score native evidence independently; controller's success flag is unused."""
    rows=_trace(result);event_scores=_event_scores(result,rows);final=rows[-250:]
    legacy=all(_predicate(row,False) for row in final)
    strict=all(_predicate(row,True) for row in final)
    retrieved=any(row['shoe_x_m']<=.53 and abs(row['shoe_y_m'])<=.21 for row in rows)
    case=result.get('case',{})
    insufficient=(all(key in case for key in ('shoe_x','tool_length'))
                  and _number(case['shoe_x'])+.165-_number(case['tool_length'])>.54)
    events=result.get('events',[])
    deferred=any(e.get('type')=='help_requested' and e.get('reason')=='tool_cannot_reach_far_edge' for e in events)
    scores={'schema':'dvidia.continuous-shoe-grip-score/v2','valid':result['valid'],
            'task_success':bool(result['valid'] and strict and retrieved),
            'legacy_task_success':bool(result['valid'] and legacy and retrieved),
            'retrieval_evidenced':retrieved,'legacy_retention_pass':legacy,'strict_retention_pass':strict,
            'legacy_only_claim':bool(legacy and not strict),'final_window_ticks':len(final),
            'final_window_duration_s':.5,'final_window_start_exclusive_s':rows[-1]['t_s']-.5,
            'final_window_end_s':rows[-1]['t_s'],
            'strict_passing_ticks':sum(_predicate(row) for row in final),
            'legacy_passing_ticks':sum(_predicate(row,False) for row in final),
            'world_supported_ticks':sum(row['world_shoe_force_n']>=.01 for row in final),
            'rake_supported_ticks':sum(row['tool_shoe_force_n']>=.01 for row in final),
            'other_robot_supported_ticks':sum(row['other_robot_force_n']>=.01 for row in final),
            'bilateral_contact_missing_ticks':sum(min(row['left_finger_force_n'],row['right_finger_force_n'])<=.01 for row in final),
            'height_condition_failed_ticks':sum(row['shoe_z_m']<=.035 for row in final),
            'deferral_requested':deferred,'insufficient_declared_tool_reach':bool(insufficient),
            'appropriate_deferral':bool(insufficient and deferred),'deferral_counts_as_task_success':False,
            'scope':'Synthetic continuous-retention scoring. No injury threshold, physical safety or learned policy claim.'}
    scores.update(event_scores)
    return scores


def audit_result(result):
    """Check claims against their declared predicate; return uniform strict score.

    Legacy v01 omitted non-finger support exclusion. Its correctly reported legacy
    pass stays mechanically valid even when the stricter independent score fails.
    Both policies must use ``task_success`` from this audit for the comparison.
    """
    scores=score_result(result);metrics=result.get('metrics',{})
    claim_scope=metrics.get('retention_claim_scope','legacy')
    if claim_scope not in ('legacy','strict'):raise ValueError('Unknown retention claim scope')
    claimed=metrics.get('retained_lift_success')
    expected=scores['legacy_retention_pass'] if claim_scope=='legacy' else scores['strict_retention_pass']
    if type(claimed) is not bool or claimed!=expected:raise ValueError('Retained-lift claim disagrees with dense native evidence')
    if type(result.get('success')) is not bool or result['success']!=(claimed and bool(metrics.get('retrieval_success'))):
        raise ValueError('Task-success claim disagrees with declared retention and retrieval')
    if metrics.get('retrieval_success') and not scores['retrieval_evidenced']:
        raise ValueError('Retrieval claim lacks recorded workspace entry')
    scores['claim_scope']=claim_scope
    return scores
