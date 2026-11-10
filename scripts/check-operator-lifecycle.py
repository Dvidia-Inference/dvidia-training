#!/usr/bin/env python3
"""Scripted local TCP lifecycle diagnostic; does not impersonate a person."""
import argparse
import json
from pathlib import Path
import socket

from dvidia_training import godot_demonstration_bridge as bridge
from dvidia_training import piper_action_capture as files


def check(scene,port):
    info=json.loads((Path(scene)/'scene.json').read_text())['bridge']
    trace=[]
    with socket.create_connection(('127.0.0.1',port),timeout=10) as connection:
        stream=connection.makefile('rb')
        def exchange(request):
            connection.sendall((json.dumps(request)+'\n').encode())
            line=stream.readline(2*1024*1024)
            if not line.endswith(b'\n'):raise ValueError('Truncated response.')
            response=json.loads(line);trace.append(dict(request=request,response=response));return response
        response=exchange(dict(type='hello',schema=bridge.SCHEMA,model_sha256=info['model_sha256']))
        assert response['recording']['source_kind']=='simulation_scripted_input'
        target=info['home_target']
        step=lambda seq:dict(type='step',sequence=seq,target=target,jaw=.035)
        assert exchange(step(1))['code']=='recording_status'
        assert exchange(dict(type='begin',sequence=1))['state']['time']==0.
        assert abs(exchange(step(2))['state']['time']-.02)<1e-10
        assert abs(exchange(dict(type='pause',sequence=3))['state']['time']-.02)<1e-10
        assert exchange(dict(type='ping'))['expected_sequence']==4
        assert exchange(step(4))['code']=='recording_status'
        assert abs(exchange(dict(type='resume',sequence=4))['state']['time']-.02)<1e-10
        assert abs(exchange(step(5))['state']['time']-.04)<1e-10
        first=exchange(dict(type='finish',sequence=6))
        assert first['recording']['status']=='finished' and first['recording']['steps']==2
        reset=exchange(dict(type='reset',sequence=7))
        assert reset['state']['time']==0. and reset['recording']['status']=='ready'
        assert first['recording']['output_directory']!=reset['recording']['output_directory']
        assert exchange(dict(type='begin',sequence=8))['state']['time']==0.
        assert abs(exchange(step(9))['state']['time']-.02)<1e-10
        stream.close()
    return dict(schema='dvidia.operator-lifecycle-check/v1',ok=True,trace=trace,
        teacher_queries=0,human_demonstrations=0,physical_robot_ready=False,
        scope='Scripted localhost control, pause/resume/reset and disconnect; no GUI button clicks.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8431)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=check(args.scene,args.port)
    files._json(args.output,result)
    print(json.dumps(dict(ok=result['ok'],exchanges=len(result['trace']),output=str(args.output))))


if __name__=='__main__':main()
