"""Write a self-contained, offline replay of privileged simulation state."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _vector(value: Any, label: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three finite coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise ValueError(f"{label} must contain three finite coordinates")


def save_replay(payload: dict[str, Any], path: str | Path) -> Path:
    """Save the runner's recorded trajectory without any runtime dependencies.

    Positions, target and actions are simulator values; this viewer creates no
    sensor measurements and runs no physics. It rejects nonfinite coordinates.
    """
    if payload.get("schema_version") != 1 or payload.get("task") != "CableEndReach-v0":
        raise ValueError("Unsupported replay schema or task")
    if payload.get("scope") != "simulation-only":
        raise ValueError("The replay must declare simulation-only scope")
    if payload.get("actuation") != "ideal endpoint attachment" or payload.get("observations") != "privileged simulator state":
        raise ValueError("The replay must declare ideal attachment and privileged state")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("At least one recorded episode is required")
    for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"Episode {episode_index} is not a record")
        frames = episode.get("trace")
        if not isinstance(frames, list):
            raise ValueError(f"Episode {episode_index} requires a trace list")
        if not frames:
            if episode.get("success") is not False or not episode.get("reason"):
                raise ValueError(f"Episode {episode_index} without frames must declare failure")
            continue
        previous_time = -math.inf
        for frame_index, frame in enumerate(frames):
            label = f"Episode {episode_index}, frame {frame_index}"
            time = frame.get("time")
            if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < previous_time:
                raise ValueError(f"{label} has invalid or decreasing time")
            previous_time = time
            for key in ("endpoint_position", "target"):
                _vector(frame.get(key), f"{label} {key}")
            for key in ("endpoint_velocity", "last_action"):
                if key in frame:
                    _vector(frame[key], f"{label} {key}")
            points = frame.get("rope_points")
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError(f"{label} requires a cable polyline")
            for point in points:
                _vector(point, f"{label} cable point")
    # HTML's script parser treats </script> specially even in JSON strings.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_PAGE.replace("__REPLAY_PAYLOAD__", encoded), encoding="utf-8")
    return output


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cable end reach — simulation replay</title>
<style>
:root{color-scheme:light;font-family:Georgia,"Times New Roman",serif;color:#262623;background:#fdfdfb}*{box-sizing:border-box}body{margin:0;padding:38px 24px;line-height:1.6}.sheet{max-width:1050px;margin:auto}.eyebrow,label,.metrics,.caption,.scope{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px;color:#67675f}h1{font-size:32px;line-height:1.25;font-weight:400;margin:9px 0 13px}.intro{max-width:76ch;font-size:16px;margin:0 0 25px}.scope{border-block:1px solid #d2d2c9;padding:12px 0;margin:0 0 22px}.controls{display:flex;align-items:end;flex-wrap:wrap;gap:14px;margin:18px 0}.control{display:flex;flex-direction:column;gap:5px}.episode{flex:1;min-width:200px}button,select{font:12px/1.5 ui-monospace,monospace;color:inherit;border:1px solid #b9b9b0;background:#fdfdfb;border-radius:0;padding:9px 12px;min-height:39px}button{cursor:pointer;min-width:84px}input{accent-color:#45453d}input[type=range]{width:100%;min-height:32px;cursor:pointer}.timeline{flex:1;min-width:200px}.angle{width:170px}.viewer{border:1px solid #d2d2c9;position:relative;aspect-ratio:1.7;min-height:270px}canvas{width:100%;height:100%;display:block}.legend{display:flex;gap:23px;flex-wrap:wrap;font:11px ui-monospace,monospace;margin:13px 0}.legend span::before{content:"";display:inline-block;width:17px;height:2px;background:#35352e;vertical-align:middle;margin-right:7px}.legend .target::before{height:10px;width:10px;border:1px dashed #657b66;border-radius:50%;background:none}.legend .endpoint::before{height:6px;width:6px;border-radius:50%;background:#4e6850}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;border-top:1px solid #d2d2c9;padding-top:17px;margin-top:23px}.metrics dt{color:#77776e;margin-bottom:4px}.metrics dd{margin:0;color:#262623;font-size:12px;overflow-wrap:anywhere}.caption{margin:22px 0 0;max-width:95ch}.status{font-size:13px;margin:0 0 8px}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #35352f;outline-offset:4px}noscript{display:block;padding:20px;border:1px solid #b9b9b0}@media(max-width:600px){body{padding:25px 17px}h1{font-size:27px}.viewer{aspect-ratio:1.25;min-height:250px}.metrics{grid-template-columns:1fr 1fr}.angle{width:130px}.timeline{flex-basis:100%}.caption{font-size:10px}}
</style></head><body><main class="sheet">
<p class="eyebrow">DVIDIA · local simulation lab</p><h1>Cable end reach</h1>
<p class="intro">Inspect a recorded cable trajectory, its target and the controller's commanded force. Rotate the projected view or move through the recorded timesteps.</p>
<p class="scope">Simulation only · Ideal endpoint attachment · Privileged simulator state</p>
<div class="controls"><div class="control episode"><label for="episode">Recorded episode</label><select id="episode"></select></div><button id="play" type="button" aria-label="Play recorded frames">Play</button><div class="control angle"><label for="angle">View angle</label><input id="angle" type="range" min="-180" max="180" value="-35"></div></div>
<p id="status" class="status"></p><div class="viewer"><canvas id="scene" role="img" aria-label="Projected cable, fixed anchor, endpoint and target in simulation"></canvas></div>
<div class="legend" aria-label="View legend"><span>Cable</span><span class="endpoint">Endpoint</span><span class="target">Target tolerance</span></div>
<div class="controls"><div class="control timeline"><label for="frame">Recorded timestep</label><input id="frame" type="range" min="0" max="0" value="0" step="1"></div></div>
<dl class="metrics"><div><dt>Simulated time</dt><dd id="time"></dd></div><div><dt>Target distance</dt><dd id="distance"></dd></div><div><dt>Recorded outcome</dt><dd id="outcome"></dd></div><div><dt>Endpoint position [m]</dt><dd id="position"></dd></div><div><dt>Commanded force [N]</dt><dd id="force"></dd></div><div><dt>Frame</dt><dd id="frame-label"></dd></div></dl>
<p class="caption">The cable uses authored simulator material parameters. The attachment assumes an ideal grasp and the controller sees simulator state. The grid is a coordinate reference, not a floor collider. These traces are neither measured tactile feedback nor evidence of calibrated contact, knot tying or physical robot transfer. Playback interpolates no new physical states.</p>
<noscript>This saved replay needs JavaScript to draw its already recorded frames. The original JSON remains the numerical record.</noscript>
</main><script id="replay-data" type="application/json">__REPLAY_PAYLOAD__</script>
<script>
'use strict';
const replay=JSON.parse(document.getElementById('replay-data').textContent);
const picker=document.getElementById('episode'),slider=document.getElementById('frame'),angle=document.getElementById('angle'),play=document.getElementById('play'),canvas=document.getElementById('scene'),ctx=canvas.getContext('2d');
let episodeIndex=0,playing=false,lastTick=0,playbackTime=0,boundsCache=null,animationId=null;
const field=id=>document.getElementById(id),vector=v=>v.map(n=>Number(n).toFixed(3)).join(', ');
replay.episodes.forEach((episode,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`${episode.policy || 'Controller'} · seed ${episode.seed} · ${episode.success ? 'success' : 'incomplete'}`;picker.append(option)});
function currentEpisode(){return replay.episodes[episodeIndex]}
function stop(){playing=false;if(animationId!==null)cancelAnimationFrame(animationId);animationId=null;play.textContent='Play';play.setAttribute('aria-label','Play recorded frames')}
function selectEpisode(){stop();episodeIndex=Number(picker.value);const hasFrames=currentEpisode().trace.length>0;slider.max=String(Math.max(0,currentEpisode().trace.length-1));slider.value='0';slider.disabled=!hasFrames;play.disabled=!hasFrames;angle.disabled=!hasFrames;draw()}
function draw(){
 const episode=currentEpisode(),frameIndex=Number(slider.value),frame=episode.trace[frameIndex];
 const box=canvas.getBoundingClientRect(),width=box.width,height=box.height,dpr=Math.min(window.devicePixelRatio||1,2);
 if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr)}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
 if(!frame){
  const reason=episode.reason||'No recorded trajectory';
  ctx.font='13px monospace';ctx.fillStyle='#77776e';ctx.textAlign='center';ctx.fillText('No usable trajectory recorded',width/2,height/2);ctx.textAlign='left';
  field('status').textContent=`${episode.policy||'Controller'} · seed ${episode.seed} · ${reason}`;
  field('time').textContent=Number.isFinite(episode.simulated_seconds)?`${episode.simulated_seconds.toFixed(3)} s`:'Unavailable';
  field('distance').textContent='Unavailable';field('outcome').textContent=`Incomplete · ${reason} · observation unavailable`;field('position').textContent='Unavailable';field('force').textContent='Not recorded';field('frame-label').textContent='0 / 0';
  canvas.setAttribute('aria-label',`No usable simulation trajectory: ${reason}`);return;
 }
 const yaw=Number(angle.value)*Math.PI/180,pitch=.43,cos=Math.cos(yaw),sin=Math.sin(yaw);
 const rotate=p=>[cos*p[0]-sin*p[1],-Math.sin(pitch)*(sin*p[0]+cos*p[1])-Math.cos(pitch)*p[2]];
 const tolerance=Number(replay.config?.tolerance)||.025;
 const boundsKey=episodeIndex+':'+angle.value;
 if(!boundsCache||boundsCache.key!==boundsKey){
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  const include=p=>{const r=rotate(p);minX=Math.min(minX,r[0]);maxX=Math.max(maxX,r[0]);minY=Math.min(minY,r[1]);maxY=Math.max(maxY,r[1])};
  episode.trace.forEach(f=>{f.rope_points.forEach(include);include(f.target)});
  boundsCache={key:boundsKey,minX:minX-tolerance,maxX:maxX+tolerance,minY:minY-tolerance,maxY:maxY+tolerance};
 }
 const {minX,maxX,minY,maxY}=boundsCache;
 const scale=Math.min((width-100)/Math.max(maxX-minX,.1),(height-85)/Math.max(maxY-minY,.1));
 const center=[(minX+maxX)/2,(minY+maxY)/2];
 const project=p=>{const r=rotate(p);return[width/2+(r[0]-center[0])*scale,height/2+(r[1]-center[1])*scale]};
 const line=(points,color,lineWidth,dash=[])=>{ctx.beginPath();points.forEach((p,i)=>{const q=project(p);if(i)ctx.lineTo(...q);else ctx.moveTo(...q)});ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([])};
 const anchor=frame.rope_points[0],extent=Number(replay.config?.rope_length)||.5;
 for(let k=-2;k<=6;k++){let v=k*.1;line([[anchor[0]-.2,anchor[1]+v,0],[anchor[0]+extent+.15,anchor[1]+v,0]],'#e8e8e1',1);line([[anchor[0]+v,anchor[1]-.2,0],[anchor[0]+v,anchor[1]+.6,0]],'#e8e8e1',1)}
 line(frame.rope_points,'#35352e',Math.max(2,Math.min(6,(Number(replay.config?.rope_radius)||.004)*2*scale)));
 const target=project(frame.target);ctx.beginPath();ctx.arc(...target,tolerance*scale,0,Math.PI*2);ctx.setLineDash([4,4]);ctx.strokeStyle='#657b66';ctx.lineWidth=1.2;ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#657b66';ctx.beginPath();ctx.arc(...target,2,0,Math.PI*2);ctx.fill();
 const end=project(frame.endpoint_position);ctx.fillStyle='#4e6850';ctx.beginPath();ctx.arc(...end,4,0,Math.PI*2);ctx.fill();
 const origin=project(anchor);ctx.fillStyle='#35352e';ctx.fillRect(origin[0]-3,origin[1]-3,6,6);
 ctx.font='10px monospace';ctx.fillStyle='#77776e';ctx.fillText('Fixed anchor',origin[0]+9,origin[1]-8);ctx.fillText('Target',target[0]+tolerance*scale+7,target[1]+3);
 field('status').textContent=`${episode.policy || 'Controller'} · seed ${episode.seed} · recorded ${Number(episode.simulated_seconds || episode.trace.at(-1).time).toFixed(2)} simulated seconds`;
 field('time').textContent=`${frame.time.toFixed(3)} s`;field('distance').textContent=`${Math.hypot(...frame.endpoint_position.map((n,i)=>n-frame.target[i])).toFixed(4)} m`;field('outcome').textContent=`${episode.success ? 'Success' : 'Incomplete'} · ${episode.reason || 'unspecified'}${episode.final_observation_available===false ? ' · final observation unavailable' : ''}`;field('position').textContent=vector(frame.endpoint_position);field('force').textContent=frame.last_action ? vector(frame.last_action) : 'Not recorded';field('frame-label').textContent=`${frameIndex+1} / ${episode.trace.length}`;
 canvas.setAttribute('aria-label',`Simulation cable at ${frame.time.toFixed(3)} seconds; endpoint ${vector(frame.endpoint_position)} metres; target distance ${field('distance').textContent}`);
}
function animate(timestamp){
 animationId=null;
 if(!playing)return;
 const episode=currentEpisode();let index=Number(slider.value),previous=index;
 if(lastTick)playbackTime+=(timestamp-lastTick)/1000;lastTick=timestamp;
 while(index+1<episode.trace.length&&episode.trace[index+1].time<=playbackTime)index++;
 if(index!==previous){slider.value=String(index);draw()}
 if(index>=episode.trace.length-1)stop();
 if(playing)animationId=requestAnimationFrame(animate);
}
play.addEventListener('click',()=>{if(playing)stop();else{if(Number(slider.value)>=Number(slider.max))slider.value='0';playing=true;lastTick=0;playbackTime=currentEpisode().trace[Number(slider.value)].time;play.textContent='Pause';play.setAttribute('aria-label','Pause recorded frames');draw();animationId=requestAnimationFrame(animate)}});
picker.addEventListener('change',selectEpisode);slider.addEventListener('input',()=>{stop();draw()});angle.addEventListener('input',draw);window.addEventListener('resize',draw);document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});selectEpisode();
</script></body></html>'''
