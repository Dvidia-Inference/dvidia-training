"""Fixed scene evaluation and native-contact negative controls for the placement adapter."""
from dataclasses import replace
import argparse
import json
from pathlib import Path

from .arm_runner import (authored_grounding, config_for, payload_for, rollout, write_run)
from .cli import NetworkGuard
from .skillspace import load_skillspace

SCENES = (
    ('default', [.42,-.04,.31], [.54,.10,.31]),
    ('cross_right', [.38,.09,.31], [.52,-.09,.31]),
    ('cross_left', [.48,-.12,.31], [.37,.09,.31]),
    ('forward', [.35,-.08,.31], [.54,.08,.31]),
    ('reverse', [.46,.10,.31], [.36,-.10,.31]),
    ('wide', [.30,-.15,.31], [.50,.18,.31]),
)


def main():
    parser = argparse.ArgumentParser(description='Fixed articulated-arm scene protocol with native contact controls.')
    parser.add_argument('--source', type=Path, default=Path(__file__).with_name('place_cup.skill.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/arm-v0'))
    args = parser.parse_args()
    source = args.source.read_bytes()
    document = json.loads(source)
    grounding = None if 'simulation_grounding' in document else authored_grounding(source)
    skill = load_skillspace(source, grounding)
    initial = config_for(skill)
    episodes = []
    with NetworkGuard(True) as guard:
        for index, (name, start, target) in enumerate(SCENES):
            scene = {'object_position':start,'target_position':target,'seed':400+index}
            config = config_for(skill, scene)
            placement = rollout(config, scene['seed'])
            placement['scene_id'] = name
            episodes.append(placement)
            for policy in ('idle','replay_open_jaw'):
                control = rollout(config, scene['seed'], policy, action_tape=placement['actions'])
                control['scene_id'] = name
                episodes.append(control)
            print(json.dumps({'scene':name,'placement':placement['reason'],
                              'negative_control_successes':sum(e['success'] for e in episodes[-2:])}), flush=True)
    payload = payload_for(skill, initial, episodes, guard.record())
    payload.update(protocol={'name':'frozen-six-layout-placement-v0',
                    'seeds':list(range(400,406)), 'scenes':[{'id':name,'object_position':obj,'target_position':target} for name,obj,target in SCENES],
                    'controller':'same authored controller; no parameter training',
                    'negative_control':'Replay identical joint target commands with jaws held open; no object state edits.',
                    'selection':'Scene protocol fixed in source before this evaluation; small bounded simulation study.'},
                   source_url='https://dvidia.org/packs/dvidia/place_cup/skill.json')
    payload['summary']['by_policy'] = {
        policy:{'successes':sum(e['success'] for e in episodes if e['policy']==policy),
                'episodes':sum(e['policy']==policy for e in episodes),
                'simulated_seconds':sum(e['simulated_seconds'] for e in episodes if e['policy']==policy),
                'episode_wall_seconds':sum(e['episode_wall_seconds'] for e in episodes if e['policy']==policy)}
        for policy in ('placement','idle','replay_open_jaw')}
    write_run(args.output,payload)
    args.output.joinpath('source.json').write_bytes(source)
    if grounding is not None:
        args.output.joinpath('grounding.json').write_text(json.dumps(grounding,indent=2)+'\n')


if __name__=='__main__':
    main()
