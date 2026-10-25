"""NVIDIA diffuse-map adaptation: native shoe appearance and physics invariance.

No shader graph, USD physics conversion, task policy or calibrated contact is
claimed. The single-image color check diagnoses the existing proxy observer.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import time
from pathlib import PurePosixPath

import numpy as np

MATERIALS = ('Fabric_Denim', 'Fabric_Cotton_White', 'Fabric_Polyester_Honeycomb_Green',
             'Fabric_Canvas', 'Fabric_Polyester_Safety_Pink')
PHYSICS_ARRAYS = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
    'geom_type', 'geom_bodyid', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction',
    'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_priority', 'geom_solref',
    'geom_solimp', 'geom_margin', 'geom_gap', 'jnt_type', 'jnt_axis', 'jnt_pos',
    'jnt_range', 'jnt_stiffness', 'dof_armature', 'dof_damping', 'dof_frictionloss',
    'actuator_gainprm', 'actuator_biasprm', 'actuator_gear', 'actuator_forcerange',
    'actuator_ctrlrange', 'eq_type', 'eq_data')
SOURCES = ('material_appearance_benchmark.py', 'material_appearance.py', 'asset_library.py',
           'asset_catalog.json', 'robot_profiles.py', 'robot_profiles.lock.json',
           'reference_sim.py', 'sensor_goal_observers.py', 'video_goal_bridge.py', 'footage_data.py')


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_run(root):
    """Verify the retained bytes and reported denominators; does not rerun physics."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Evidence must be a local directory')
    inventory = json.loads((root/'integrity.json').read_text())['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p != root/'integrity.json'}
    if actual != set(inventory): raise ValueError('Evidence file inventory changed')
    for name, record in inventory.items():
        parts = PurePosixPath(name).parts
        if not parts or name.startswith('/') or '\\' in name or '..' in parts:
            raise ValueError('Unsafe evidence path')
        path = root/name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Evidence symlinks are not allowed')
        if path.stat().st_size != record['bytes'] or _sha(path) != record['sha256']:
            raise ValueError('Evidence hash mismatch: '+name)
    report = json.loads((root/'report.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if report['protocol_sha256'] != _sha(root/'protocol.json'):
        raise ValueError('Protocol binding changed')
    if [r['id'] for r in report['rows']] != ['original', *protocol['materials']]:
        raise ValueError('Scheduled materials changed')
    for key, expected in dict(scheduled=len(report['rows']),
            completed=sum(r['status'] == 'completed' for r in report['rows']),
            unchanged_mechanics=sum(r['mechanics_equal'] for r in report['rows']),
            identical_native_trajectories=sum(r['trajectory_equal'] for r in report['rows'])).items():
        if report[key] != expected: raise ValueError('Invalid denominator: '+key)
    for name, digest in protocol['source_sha256'].items():
        if _sha(root/'sources'/name) != digest: raise ValueError('Source binding changed')
    from .material_appearance import load_appearance
    for material in protocol['materials']:
        if (root/'materials'/material).exists(): load_appearance(root/'materials'/material)
    return dict(verified_files=len(inventory), scheduled=report['scheduled'],
                completed=report['completed'], physics_rerun=False)


def mechanics(model):
    """Snapshot the explicitly listed mechanics, excluding appearance arrays."""
    arrays = {name: getattr(model, name).copy() for name in PHYSICS_ARRAYS}
    arrays['gravity'] = model.opt.gravity.copy()
    arrays['timestep'] = np.array(model.opt.timestep)
    arrays['dimensions'] = np.array([model.nq, model.nv, model.nu, model.neq, model.nbody])
    return arrays


def mechanics_comparison(before, after):
    return dict(equal=all(np.array_equal(before[n], after[n]) for n in before),
                changed_arrays=[n for n in before if not np.array_equal(before[n], after[n])])


def simulate(model):
    import mujoco as mj
    from .asset_library import initialize
    data = initialize(model)
    trace = []
    for i in range(1001):
        if i % 250 == 0:
            trace.append(dict(t_s=float(data.time), qpos=data.qpos.tolist(),
                              qvel=data.qvel.tolist(), contacts=int(data.ncon)))
        if i < 1000: mj.mj_step(model, data); mj.mj_forward(model, data)
    if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
        raise RuntimeError('Native shoe settling warning or invalid state')
    return data, trace


def _png(path, rgb):
    from .image_goal_scenes import _png as encode
    with path.open('xb') as stream: stream.write(encode(rgb))


def render_shoes(model, data):
    import mujoco as mj
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    camera = mj.MjvCamera(); camera.lookat[:] = [.37, .42, .06]
    camera.azimuth = 135.; camera.elevation = -35.; camera.distance = 1.05
    with mj.Renderer(model, 480, 640) as renderer:
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()


def color_fixture(appearance):
    """One actual rendered image of a supported cube; no temporal claim."""
    import mujoco as mj
    from .material_appearance import apply_to_spec
    from .sensor_goal_observers import CameraCalibration, render_overhead_rgb
    from .video_goal_bridge import PlanarVideoCalibration, _observe_frame
    spec = mj.MjSpec()
    spec.option.timestep = .002
    spec.worldbody.add_geom(name='floor', type=mj.mjtGeom.mjGEOM_PLANE,
                            size=[2, 2, .1], rgba=[.8, .8, .8, 1])
    body = spec.worldbody.add_body(name='cube', pos=[.36, 0, .021])
    body.add_freejoint()
    body.add_geom(name='cube', type=mj.mjtGeom.mjGEOM_BOX, size=[.02]*3,
                  mass=.04, rgba=[.1, .65, .2, 1], friction=[.8, .005, .0001])
    if appearance is not None: apply_to_spec(spec, appearance, ['cube'])
    model = spec.compile(); data = mj.MjData(model)
    for _ in range(250): mj.mj_step(model, data)
    mj.mj_forward(model, data)
    cal = PlanarVideoCalibration()
    rgb = render_overhead_rgb(model, data, calibration=CameraCalibration(width=cal.width,
        height=cal.height, center_xy_m=cal.center_xy_m, vertical_span_m=cal.vertical_span_m))
    return rgb, _observe_frame(rgb, 0., cal), data.qpos.copy()


def run(archive, assets, output, *, materials=MATERIALS):
    from .material_appearance import prepare_appearance, catalog_archive
    from .asset_library import build_scene
    from .robot_profiles import verify_assets
    import mujoco as mj
    output = Path(output)
    if output.exists(): raise ValueError('Use a fresh experiment directory')
    # Verify source before creating any import output or native model.
    available = catalog_archive(archive)
    if not isinstance(materials, (list, tuple)) or not 1 <= len(materials) <= 13 or len(set(materials)) != len(materials):
        raise ValueError('Select one to thirteen distinct fabric IDs')
    asset_provenance = verify_assets('piper', assets)
    output.mkdir(parents=True)
    protocol = dict(schema='dvidia.material-appearance-protocol/v1', materials=list(materials),
        robot='piper', shoes=['trainer-left', 'trainer-right'], timestep_s=.002,
        physics_s=2., compared_arrays=list(PHYSICS_ARRAYS)+['gravity', 'timestep', 'dimensions'],
        source_archive_sha256=_sha(archive), source_archive_catalog=available,
        robot_asset_provenance=asset_provenance, pixel_fixture='One static supported rigid cube per appearance; not a video/task-success test.',
        source_sha256={name: _sha(Path(__file__).with_name(name)) for name in SOURCES},
        platform=dict(os=platform.platform(), numpy=np.__version__, mujoco=mj.__version__),
        rules='Original plus all selected appearances count; failures remain. No outcome-based replacement.',
        physical_properties='Keep original authored mass/friction/inertia/collision/contact settings. No material-name-to-physics mapping.')
    _write(output/'protocol.json', protocol)
    (output/'sources').mkdir()
    for name in SOURCES: (output/'sources'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    baseline_arrays = baseline_trace = baseline_rgb = baseline_cube_pose = None
    rows = []; start = time.perf_counter()
    for identifier in ('original', *materials):
        row = dict(id=identifier, status='failed', error=None, mechanics_equal=False,
                   trajectory_equal=False, single_image_color_observation=None)
        began = time.perf_counter()
        print(json.dumps(dict(stage='appearance', material=identifier)), flush=True)
        try:
            directory = None
            if identifier != 'original':
                directory = output/'materials'/identifier
                prepare_appearance(archive, identifier, directory, texture_size=256)
            model = build_scene('piper', assets, shoe_ids=['trainer-left', 'trainer-right'], appearance=directory)
            arrays = mechanics(model)
            if baseline_arrays is None: baseline_arrays = arrays
            row['mechanics'] = mechanics_comparison(baseline_arrays, arrays)
            row['mechanics_equal'] = row['mechanics']['equal']
            data, trace = simulate(model)
            if baseline_trace is None: baseline_trace = trace
            row['trajectory_equal'] = trace == baseline_trace
            row['native_trace'] = trace
            rgb = render_shoes(model, data)
            _png(output/(identifier+'-shoes.png'), rgb)
            if baseline_rgb is None: baseline_rgb = rgb
            row['mean_absolute_pixel_change'] = float(np.abs(rgb.astype(float)-baseline_rgb).mean())
            row['texture_bytes_native'] = int(model.tex_data.nbytes)
            row['shoe_image'] = identifier+'-shoes.png'
            cube, observation, pose = color_fixture(directory)
            if baseline_cube_pose is None: baseline_cube_pose = pose
            row['cube_pose_equal'] = bool(np.array_equal(pose, baseline_cube_pose))
            row['single_image_color_observation'] = observation
            row['cube_image'] = identifier+'-cube.png'
            _png(output/row['cube_image'], cube)
            row['status'] = 'completed'
        except Exception as exc:
            row['error'] = dict(type=type(exc).__name__, message=str(exc)[:500])
        row['wall_seconds'] = time.perf_counter()-began
        rows.append(row)
    report = dict(schema='dvidia.material-appearance-benchmark/v1',
        protocol_sha256=_sha(output/'protocol.json'), scheduled=len(materials)+1,
        completed=sum(r['status'] == 'completed' for r in rows), rows=rows,
        unchanged_mechanics=sum(r['mechanics_equal'] for r in rows),
        identical_native_trajectories=sum(r['trajectory_equal'] for r in rows),
        wall_seconds=time.perf_counter()-start, motor_policy_trained=False,
        physics_calibrated=False, original_material_graph_evaluated=False,
        limitations=['Diffuse/albedo approximation; no MaterialX/OpenPBR graph, normal, roughness, fuzz or nonvisual sensor response reproduced.',
            'Native invariance is against the existing authored rigid shoe proxies, not measured shoe behavior.',
            'The static cube color detector diagnostic has no task, identity, pose-accuracy or robustness claim.',
            'Appearance variety creates potential training data; no model was retrained and no training benefit was measured.'])
    _write(output/'report.json', report)
    _html(output, report)
    _write(output/'integrity.json', dict(files={str(p.relative_to(output)):dict(bytes=p.stat().st_size, sha256=_sha(p))
        for p in sorted(output.rglob('*')) if p.is_file()}))
    return report


def _html(output, report):
    cards = ''
    for row in report['rows']:
        state = (row.get('single_image_color_observation') or {}).get('status', 'not evaluated')
        cards += f'<section><h2>{escape(row["id"])}</h2><p>Mechanics unchanged: {row["mechanics_equal"]} · Native trajectory identical: {row["trajectory_equal"]} · Green-proxy observer: {escape(state)}</p>'
        if row.get('shoe_image'): cards += f'<img src="{escape(row["shoe_image"], quote=True)}" alt="Native 3D shoe scene with {escape(row["id"], quote=True)} appearance">'
        if row.get('cube_image'): cards += f'<img class="cube" src="{escape(row["cube_image"], quote=True)}" alt="Single-image color diagnostic">'
        if row['error']: cards += '<p>'+escape(str(row['error']))+'</p>'
        cards += '</section>'
    (output/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DVIDIA · NVIDIA fabric appearance</title>
<style>body{{max-width:960px;margin:44px auto;padding:0 24px;background:#faf9f5;color:#252c2b;font:18px/1.6 Georgia,serif}}h1,h2{{font-family:system-ui,sans-serif}}h1{{line-height:1.2}}section{{border-top:1px solid #ddd;margin-top:32px}}img{{width:min(100%,640px)}}img.cube{{width:min(100%,280px)}}a{{color:#315b4d}}</style>
<p>DVIDIA research · external appearance pilot</p><h1>Real fabric textures, separately tested mechanics</h1>
<p>NVIDIA v0.2.0 Fabric diffuse maps are adapted to MuJoCo. Shoe geometry, mass, collision and contact remain our original rigid assumptions. No new motor policy or calibrated fabric physics is claimed.</p>
<p>{report['completed']}/{report['scheduled']} cases completed; {report['unchanged_mechanics']}/{report['scheduled']} mechanics comparisons matched; {report['identical_native_trajectories']}/{report['scheduled']} two-second trajectories matched.</p>
<p><a href="report.json">All results</a> · <a href="protocol.json">Frozen protocol</a> · <a href="integrity.json">File hashes</a>. The cube panels test only whether the existing green-color rule sees a candidate.</p>{cards}</html>''')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.archive, args.assets, args.output)
    print(json.dumps({k: result[k] for k in ('scheduled', 'completed', 'unchanged_mechanics', 'identical_native_trajectories', 'wall_seconds')}))


if __name__ == '__main__': main()
