"""Mask-aware v2 exports, separate from the intentionally strict v1 exporter.

Only pre-action measurements and separately sealed perception enter inputs.
Teacher queries are explicit supervision on mixed trajectories; they are never
relabeled as commands that the producer actually executed.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import camera_action_model as model


def _document(root, reference):
    from . import action_demonstration as tape
    body = tape._ordinary_bytes(root / str(tape._relative(reference['path'])), 2*1024*1024)
    if reference['bytes'] != len(body) or reference['sha256'] != sha256(body).hexdigest():
        raise ValueError('Learning sidecar bytes no longer match their sealed reference.')
    return model.core._decode(body)


def export_episode(directory, *, blind=False, label_source='issued'):
    """Read and verify one sealed episode; no rendering, physics or relabeling.

    ``issued`` uses teacher-only producer commands. ``teacher_query`` requires
    mixed control and labels each pre-action snapshot using its one separate
    teacher query, whether the teacher or student actually controlled that step.
    Interrupted complete prefixes remain exportable; a fresh t=0 anchor is
    always required. Source selection is the experiment orchestrator's job.
    """
    from . import camera_action_capture as capture
    root = Path(directory)
    recording = capture.load_rollout(root)
    recording_body = (root/'recording.json').read_bytes()
    if recording_body.strip() != model.canonical(recording):
        raise ValueError('Recording changed after rollout verification.')
    if label_source not in ('issued', 'teacher_query'):
        raise ValueError('Choose issued or teacher_query labels explicitly.')
    expected = 'teacher' if label_source == 'issued' else 'mixed'
    if recording.get('control_mode') != expected:
        raise ValueError('Issued supervision requires teacher control; queried correction supervision requires mixed control.')
    if not recording['steps']:
        raise ValueError('No complete pre-action intervals are available.')
    encoder = model.CameraActionFeatures(blind=blind)
    previous_applied = recording['initial_applied_targets']
    features, targets, perception_hashes, query_hashes, authority = [], [], [], [], []
    for step in recording['steps']:
        perception = _document(root, step['perception'])
        context = dict(snapshot=step['before'], estimate=perception['estimate'],
            previous_applied=previous_applied, goal_position_m=recording['goal_position_m'])
        features.append(encoder(context))
        perception_hashes.append(step['perception']['sha256'])
        if label_source == 'issued':
            if step['command']['authority'] != 'teacher':
                raise ValueError('Issued imitation labels must actually come from the teacher.')
            target = step['command']['issued']
        else:
            reference = step.get('teacher_query')
            if reference is None:
                raise ValueError('A separately sealed teacher query is required for every correction label.')
            query = _document(root, reference)
            if query.get('origin') != 'privileged_simulator_teacher':
                raise ValueError('Unknown teacher-query supervision origin.')
            target = query['action']
            query_hashes.append(reference['sha256'])
        targets.append(model.action_vector(target))
        authority.append(step['command']['authority'])
        previous_applied = step['applied']['ramped_targets']
    source_group = recording['provenance']['source_group']
    source = dict(source_group=source_group, case_id=recording['provenance']['case_id'],
        recording_sha256=sha256(recording_body).hexdigest(),
        integrity_sha256=sha256((root/'integrity.json').read_bytes()).hexdigest(),
        calibration_sha256=recording['binding']['calibration_sha256'],
        tracker_sha256=recording['tracker_sha256'], label_source=label_source,
        sample_count=len(features), perception_references_sha256=model.digest(perception_hashes),
        teacher_query_references_sha256=model.digest(query_hashes),
        actual_authority_sha256=model.digest(authority))
    sources = [source]
    provenance = dict(feature_contract_sha256=model.FEATURE_CONTRACT_SHA256, blind=blind,
        episodes=sources, source_identity_sha256=model.digest(sources),
        export_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        label_definition='absolute issued teacher targets' if label_source == 'issued'
            else 'separate teacher query suggestions; actual control authority retained independently')
    return dict(schema=model.DATASET_SCHEMA, features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float64), groups=[source_group]*len(features), provenance=provenance)


def combine_datasets(datasets):
    """Concatenate already exported episodes once; retain whole-start group IDs."""
    if type(datasets) not in (list, tuple) or not datasets:
        raise ValueError('At least one exported dataset is required.')
    blind = datasets[0]['provenance']['blind']
    features, targets, groups, sources, identities = [], [], [], [], set()
    for dataset in datasets:
        x, y, g, p, _ = model._dataset(dataset, blind=blind)
        if (p.get('blind') is not blind or p.get('feature_contract_sha256') != model.FEATURE_CONTRACT_SHA256
                or p.get('source_identity_sha256') != model.digest(p.get('episodes'))):
            raise ValueError('Combined exports must have the same exact feature contract and feedback masking.')
        for episode in p['episodes']:
            identity = episode['recording_sha256']
            if identity in identities:
                raise ValueError('An episode cannot be duplicated or relabeled within one training dataset.')
            identities.add(identity)
            sources.append(episode)
        features.append(x); targets.append(y); groups.extend(g)
    if len(groups) > 1_000_000:
        raise ValueError('Combined training data exceeds its bounded sample budget.')
    provenance = dict(feature_contract_sha256=model.FEATURE_CONTRACT_SHA256, blind=blind,
        episodes=sources, source_identity_sha256=model.digest(sources),
        export_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        label_definition='per-episode explicit issued teacher targets or separately sealed teacher query suggestions')
    return dict(schema=model.DATASET_SCHEMA, features=np.concatenate(features), targets=np.concatenate(targets),
        groups=groups, provenance=provenance)


def blind_dataset(dataset):
    """Mask a verified full export without rereading its potentially large frames."""
    x, y, groups, provenance, _ = model._dataset(dataset, blind=False)
    if provenance.get('blind') is not False:
        raise ValueError('Blind masking expects a full-feedback exported dataset.')
    x = x.copy(); x[:, model.LIVE_FEATURE_START:] = 0.
    provenance = deepcopy(provenance); provenance['blind'] = True
    return dict(schema=model.DATASET_SCHEMA, features=x, targets=y.copy(),
        groups=groups, provenance=provenance)
