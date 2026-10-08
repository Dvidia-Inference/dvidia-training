"""Public DVIDIA metadata installation; no downloaded code is executed."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .skillspace import inspect_skillspace, load_skillspace

MAX_SOURCE_BYTES = 128 * 1024
PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
POLICY_ORIGIN = 'locally authored articulated-arm placement controller; source metadata is not a trained policy'
RUNTIME_SOURCE_FILES = ('arm_env.py', 'arm_policy.py', 'arm_runner.py', 'env.py', 'skillspace.py', 'cli.py', 'installer.py')
_LOADED_SOURCE_HASHES = {name: sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in RUNTIME_SOURCE_FILES}


def runtime_contract() -> dict:
    """Pin the local executable and native dependencies, separately from metadata."""
    import mujoco
    import numpy
    root = Path(__file__).parent
    files = {name: sha256((root/name).read_bytes()).hexdigest() for name in RUNTIME_SOURCE_FILES}
    if files != _LOADED_SOURCE_HASHES:
        raise ValueError('Runtime source changed while this process was running. Restart the lab before installation or execution.')
    return {'kind': 'authored-simulation-runtime', 'files_sha256': files,
            'dependencies': {'mujoco': mujoco.__version__, 'numpy': numpy.__version__},
            'action_contract': 'six joint targets in radians; symmetric gripper width in metres'}


def _identity(raw, grounding, runtime):
    binding = json.dumps({'grounding': grounding, 'runtime': runtime}, sort_keys=True, allow_nan=False)
    return sha256(raw + binding.encode()).hexdigest()[:24]


def resolve_skill_url(value: str) -> str:
    """Resolve an observed public store route to its explicit pack envelope."""
    if not isinstance(value, str) or len(value) > 1000:
        raise ValueError('Supply a public https://dvidia.org/store/owner/skill link.')
    parts = urlsplit(value.strip())
    if (parts.scheme != 'https' or parts.netloc not in ('dvidia.org', 'www.dvidia.org')
            or parts.query or parts.fragment):
        raise ValueError('Use a public DVIDIA HTTPS store or skill.json link without a query or fragment.')
    path = parts.path.rstrip('/').split('/')[1:]
    if len(path) == 3 and path[0] == 'store':
        owner, skill = path[1:]
    elif len(path) == 4 and path[0] == 'packs' and path[3] == 'skill.json':
        owner, skill = path[1:3]
    else:
        raise ValueError('This link is not a public skill release. Device-local Skillspace plans need a JSON export.')
    if not PART.fullmatch(owner) or not PART.fullmatch(skill):
        raise ValueError('The owner or skill name in this link is invalid.')
    return f'https://dvidia.org/packs/{owner}/{skill}/skill.json'


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Skill download redirected. Use its direct public DVIDIA JSON link.')


def fetch_skill_url(value: str) -> tuple[bytes, str]:
    url = resolve_skill_url(value)
    opener = build_opener(_NoRedirect())
    request = Request(url, headers={'Accept': 'application/json', 'User-Agent': 'DVIDIA-SimLab/0.2'})
    with opener.open(request, timeout=12) as response:
        data = response.read(MAX_SOURCE_BYTES + 1)
    if not data or len(data) > MAX_SOURCE_BYTES:
        raise ValueError('Skill source is empty or exceeds the 128 KiB metadata limit.')
    inspect_skillspace(data)
    return data, url


def install_source(data: bytes | str, grounding: dict, directory: Path,
                   *, source_url: str | None = None) -> dict:
    if source_url is not None:
        source_url = resolve_skill_url(source_url)
    raw = data.encode('utf8') if isinstance(data, str) else data
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_SOURCE_BYTES:
        raise ValueError('Supply at most 128 KiB of Skillspace JSON.')
    inspect_skillspace(raw)
    source = json.loads(raw)
    inline = 'simulation_grounding' in source
    if inline:
        grounding = source['simulation_grounding']
    skill = load_skillspace(raw, None if inline else grounding)
    runtime = runtime_contract()
    identity = _identity(raw, grounding, runtime)
    record = {'schema_version': 2, 'installation_id': identity, 'runtime': runtime,
              'status': 'installed_simulation_adapter', 'scope': 'simulation-only',
              'source_url': source_url, 'source_sha256': sha256(raw).hexdigest(),
              'source': source, 'grounding': grounding, 'grounding_mode': 'inline' if inline else 'sidecar',
              'skill': skill.to_dict(),
              'policy_origin': POLICY_ORIGIN,
              'physical_robot_ready': False}
    target = Path(directory) / identity
    target.mkdir(parents=True, exist_ok=True)
    (target / 'source.json').write_bytes(raw)
    (target / 'installation.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return record


def read_installation(directory: Path, identity: str) -> dict:
    if not isinstance(identity, str) or not re.fullmatch(r'[a-f0-9]{24}', identity):
        raise ValueError('Choose an installed skill.')
    root = Path(directory) / identity
    record = json.loads((root / 'installation.json').read_text(encoding='utf8'))
    source = (root / 'source.json').read_bytes()
    fields = {'schema_version','installation_id','status','scope','source_url','source_sha256','source',
              'grounding','grounding_mode','skill','policy_origin','physical_robot_ready','runtime'}
    if type(record) is not dict or set(record) != fields:
        raise ValueError('Installation envelope is invalid. Install this release again.')
    for key, value in {'schema_version':2,'status':'installed_simulation_adapter','scope':'simulation-only',
                       'policy_origin':POLICY_ORIGIN,'physical_robot_ready':False}.items():
        if type(record[key]) is not type(value) or record[key] != value:
            raise ValueError('Installation provenance is invalid. Install this release again.')
    inline = 'simulation_grounding' in json.loads(source)
    if record['grounding_mode'] != ('inline' if inline else 'sidecar'):
        raise ValueError('Installation grounding mode is invalid.')
    if inline and record['grounding'] != json.loads(source)['simulation_grounding']:
        raise ValueError('Installed inline grounding does not match its source.')
    if record['source_url'] is not None and resolve_skill_url(record['source_url']) != record['source_url']:
        raise ValueError('Installation source URL must be its canonical public JSON link.')
    if record.get('installation_id') != identity or sha256(source).hexdigest() != record.get('source_sha256'):
        raise ValueError('Installed source integrity check failed. Install this release again.')
    skill = load_skillspace(source, None if record.get('grounding_mode') == 'inline' else record['grounding'])
    if skill.to_dict() != record['skill'] or json.loads(source) != record['source']:
        raise ValueError('Installed grounding integrity check failed. Install this release again.')
    if record['runtime'] != runtime_contract():
        raise ValueError('Installed controller or physics dependency changed. Install again to review the new runtime.')
    expected = _identity(source, record['grounding'], record['runtime'])
    if identity != expected:
        raise ValueError('Installation identity does not match its source, grounding and executable runtime.')
    return record
