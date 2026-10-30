"""Guarded publication of the sealed PiPER native-capture source preview.

Only publish reads the named macOS Keychain credential. Stage/check/verify use
anonymous reads. The shared Space receives additions under one new namespace;
all existing Git/LFS objects are preserved through exact parent-CAS commits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha1, sha256
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile

from publish import PublicationError, client


SCHEMA = "dvidia.piper-action-capture-hub-publication/v1"
REPOS = {"dataset": "Dvidia/piper-action-capture-v01", "space": "Dvidia/robot-simulation-lab"}
NAMESPACE = "action-capture/v01/"
OID = re.compile(r"[a-f0-9]{40}\Z")
MAX_FILE = 128*1024*1024
MAX_PLAN = 4*1024*1024
IMPLEMENTATION = (Path(__file__).name, "publish.py", "prepare.py")


def digest(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() and str(p) not in ("/var", "/tmp") for p in (path, *path.parents)):
        raise PublicationError("Publication paths cannot traverse artifact symlinks.")
    return path.resolve()


def relative(name):
    if not isinstance(name, str):
        raise PublicationError("Expected an artifact-relative path.")
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute() or str(path) != name
            or any(part in (".", "..", "") or part.startswith(".") and name != ".gitattributes" for part in path.parts)
            or any(ord(char) < 32 for char in name)):
        raise PublicationError("Expected a plain artifact-relative path.")
    return path


def read(path, limit=MAX_FILE):
    path = plain(path)
    if not path.is_file():
        raise PublicationError("Missing publication file.")
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise PublicationError("Publication file exceeds its byte limit.")
    return raw


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()


def load(path):
    return json.loads(read(path, MAX_PLAN))


def atomic(path, value, exclusive=False):
    path = plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".writing")
    if exclusive and path.exists():
        raise PublicationError("Publication evidence already exists.")
    with temporary.open("xb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def implementation():
    return {name: digest(read(Path(__file__).with_name(name))) for name in IMPLEMENTATION}


def scan(raw):
    if re.search(rb"(?:hf_|gh[pousr]_|apikey_)[A-Za-z0-9_]{24,}", raw):
        raise PublicationError("Credential-shaped bytes found in a public payload.")


def files(root):
    root = plain(root)
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PublicationError("Public payload contains a link or special file.")
        if path.is_file():
            name = path.relative_to(root).as_posix(); relative(name)
            raw = read(path); scan(raw)
            result.append({"path": name, **digest(raw), "representation": "lfs" if name.endswith((".gz", ".zip", ".whl")) else "git"})
    if not 1 <= len(result) <= 512 or sum(item["bytes"] for item in result) > 160*1024*1024:
        raise PublicationError("Payload inventory exceeds the fixed bounds.")
    return result


def public_client():
    os.environ.update(HF_DEBUG="0", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1")
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi
    return HfApi(endpoint="https://huggingface.co", token=False)


def download(repo, name, kind, revision, cache, force_download=False):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, name, repo_type=kind, revision=revision, token=False,
        endpoint="https://huggingface.co", cache_dir=str(cache),force_download=force_download)
    # Hub cache links are trusted download outputs, never upload inputs.
    with Path(path).open('rb') as stream:raw=stream.read(MAX_FILE+1)
    if len(raw)>MAX_FILE:raise PublicationError('Anonymous object exceeds the bounded download size.')
    return raw


def tree(api, kind, revision=None):
    info = api.repo_info(REPOS[kind], repo_type=kind, revision=revision, files_metadata=True, token=False)
    if info.private is not False or not OID.fullmatch(info.sha or "") or kind == "space" and getattr(info, "sdk", None) != "static":
        raise PublicationError("Expected the fixed public repository and static Space.")
    result = {}
    for item in info.siblings:
        relative(item.rfilename)
        if (item.rfilename in result or not isinstance(item.size, int) or item.size < 0
                or not OID.fullmatch(item.blob_id or "")
                or item.lfs and not re.fullmatch(r"[a-f0-9]{64}", item.lfs.sha256 or "")):
            raise PublicationError("Public repository metadata lacks exact object identities.")
        result[item.rfilename] = {"bytes": item.size, "blob": item.blob_id,
                                  "lfs_sha256": item.lfs.sha256 if item.lfs else None}
    return {"commit": info.sha, "tree": result}


def compare(actual, expected):
    if set(actual) != set(expected):
        raise PublicationError("Public file roster differs from the reviewed tree.")
    for name, want in expected.items():
        got = actual[name]
        if (got["bytes"] != want["bytes"] or got["lfs_sha256"] != want["lfs_sha256"]
                or want["blob"] is not None and got["blob"] != want["blob"]):
            raise PublicationError("Public Git/LFS object identity differs from the reviewed tree.")


def expected_tree(row, baseline, count=None):
    result = dict(baseline)
    selected = row["files"] if count is None else row["files"][:count]
    for item in selected:
        raw = read(Path(row["root"])/relative(item["path"]))
        if digest(raw) != {key: item[key] for key in ("bytes", "sha256")}:
            raise PublicationError("Staged bytes changed.")
        result[item["path"]] = {"bytes": len(raw), "blob": sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() if item["representation"] == "git" else None,
                                "lfs_sha256": item["sha256"] if item["representation"] == "lfs" else None}
    return result


def _sealed(root, maximum_files):
    root=plain(root);raw=read(root/"manifest.json",MAX_PLAN);manifest=json.loads(raw)
    expected={**manifest["files"],"manifest.json":digest(raw)};actual={}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not(path.is_file() or path.is_dir()):raise PublicationError("Sealed inputs cannot contain links or special files.")
        if path.is_file():actual[path.relative_to(root).as_posix()]=digest(read(path))
    if len(actual)>maximum_files or actual!=expected:raise PublicationError("Sealed input membership or bytes changed.")
    return raw,manifest


def _inputs(kit,web):
    kit,web=plain(kit),plain(web)
    if kit.name!="piper-action-capture-public02-20261030" or web.name!="piper-action-capture-web02-20261030":
        raise PublicationError("Only the reviewed display-corrected v02 local exports may be staged.")
    manifest_raw,manifest=_sealed(kit,12000);web_raw,web_manifest=_sealed(web,64)
    summary=json.loads(read(kit/"summary.json"));cases=summary.get("cases",[])
    if ([x.get("id") for x in cases]!=["nominal-full","nominal-repeat","interrupted-prefix"]
            or [x.get("steps") for x in cases]!=[1000,1000,2]
            or [x.get("snapshots") for x in cases]!=[1001,1001,3]
            or summary.get("missing_tracker_channels")!=4 or summary.get("trained_skill") is not False
            or summary.get("physical_robot_ready") is not False or summary.get("preview",{}).get("version")!="0.1.3.dev1"):
        raise PublicationError("Expected the exact capture milestone and unqualified source-preview scope.")
    archive=kit.with_suffix(".zip");archive_raw=read(archive);archive_receipt=load(kit.with_suffix(".receipt.json"))
    if archive_receipt.get("manifest")!=digest(manifest_raw) or archive_receipt.get("archive")!={"name":archive.name,**digest(archive_raw)}:
        raise PublicationError("Source archive differs from the sealed complete-kit receipt.")
    expected={**manifest["files"],"manifest.json":digest(manifest_raw)}
    with zipfile.ZipFile(archive) as zipped:
        if len(zipped.infolist())!=len(expected) or set(zipped.namelist())!=set(expected):raise PublicationError("Archive roster differs from the complete kit.")
        for item in zipped.infolist():
            if item.file_size!=expected[item.filename]["bytes"] or item.file_size>16*1024*1024:
                raise PublicationError("Archive member size differs from its source seal.")
            if digest(zipped.read(item))!=expected[item.filename]:raise PublicationError("Archive member differs from its source bytes.")
    derivation=json.loads(read(web/"derivation-receipt.json"))
    if (derivation.get("source_kit_manifest")!=digest(manifest_raw)
            or [e.get("input_frame_count")for e in derivation.get("episodes",[])]!=[1001,1001,3]
            or any(int(e["ffprobe"]["nb_read_frames"])!=e["input_frame_count"] for e in derivation["episodes"])):
        raise PublicationError("Compact display does not derive from every recorded RGB boundary.")
    wheel="dvidia_training-0.1.3.dev1-py3-none-any.whl";wheel_raw=read(kit/"preview"/wheel)
    if digest(wheel_raw)!={"bytes":5494221,"sha256":"4ff5542d104c9b35f486fff80d4d2eced529be7a4295391535bccf459cff07bb"}:
        raise PublicationError("Source-preview wheel differs from the installed-smoke proof.")
    card=("---\nlicense: mit\nlanguage:\n- en\ntags:\n- robotics\n- simulation\n- mujoco\n- agilex-piper\n- dvidia\npretty_name: DVIDIA PiPER action capture v0.1\nsize_categories:\n- n<1K\n---\n\n# DVIDIA PiPER action capture v0.1\n\n"
        "A capture engineering milestone: actual rendered RGB-D, measured native encoders and authored actuator commands. "
        "Two 20-second runs and one interrupted prefix retain 2,005 snapshots and 4,010 RGB/depth files. "
        "The two full runs repeat one fixed start/controller/seed; they are not independent examples or a generalized success rate.\n\n"
        "[Inspect the recorded capture](https://dvidia-robot-simulation-lab.static.hf.space/action-capture/v01/index.html) · "
        "[Research](https://research.dvidia.org/papers/native-action-capture-v01/) · "
        "[Source preview release](https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-action-capture-v0.1-preview)\n\n"
        "Download `piper-action-capture-v01.zip`, extract it and run `python -m http.server 8000` for the complete local PNG/NPY report. "
        "The small online viewer uses lossy RGB-derived MP4; original arrays, every motor label and independent audits remain in the ZIP. "
        "Native task flags remain separate from the recordings' unverified/unknown task results. "
        "Four object-tracker channels remain missing/null, so strict learning-target export is refused. "
        "CGL depth precision warnings remain explicit; exact repetition is not depth accuracy.\n\n"
        "The included wheel is source preview **0.1.3.dev1**, distinct from stable data-only SDK **0.1.2**. "
        "Its isolated installed smoke collected five native intervals/twelve frame files using existing pinned assets. "
        "Optional native dependencies and a working OpenGL context are required for capture; no minimum hardware or hard real-time claim. "
        "Pillow 12.0.0 is additionally needed only to reproduce the independent image-decoding audit.\n\n"
        "DVIDIA source/evidence and PiPER render derivatives: MIT. The preview wheel includes MuJoCo/WASM and other dependencies retaining "
        "Apache-2.0, MIT and their original notices; the card's MIT label does not relicense them. "
        "No operator corrections, camera pose tracker, trained skill, human-video action recovery or physical robot qualification. "
        "Original development refusal/smoke evidence is separately retained, outside the three registered cases.\n\n"
        "DVIDIA research · [@iammrriver](https://x.com/iammrriver) · hello@dvidia.org.\n").encode()
    dataset={"README.md":card,".gitattributes":b"*.zip filter=lfs diff=lfs merge=lfs -text\n*.whl filter=lfs diff=lfs merge=lfs -text\n",
        "piper-action-capture-v01.zip":archive_raw,wheel:wheel_raw,"kit-manifest.json":manifest_raw,
        "archive-receipt.json":encoded(archive_receipt),"web-manifest.json":web_raw}
    for name in ("summary.json","protocol.json","paper.md","LICENSE","derivation-receipt.json"):
        dataset[name]=read((web if name=="derivation-receipt.json" else kit)/name)
    for name in ("guide.md","installed-smoke.json","wheel-manifest.json","requirements-installed-test.txt"):
        dataset[name]=read(kit/("docs/piper-action-capture.md" if name=="guide.md" else "preview/"+name))
    for name in web_manifest["files"]:
        if name.startswith(("audits/","licenses/")):dataset[name]=read(web/name)
    space={NAMESPACE+name:read(web/name) for name in web_manifest["files"]};space[NAMESPACE+"manifest.json"]=web_raw
    evidence={"kit_manifest":digest(manifest_raw),"kit_archive":digest(archive_raw),"web_manifest":digest(web_raw),"preview_wheel":digest(wheel_raw)}
    return dataset,space,evidence


def stage(kit,web,output,*,api=None,downloader=None):
    output=plain(output)
    if output.exists():raise PublicationError("Use a fresh publication-plan directory.")
    dataset_files,space_files,evidence=_inputs(kit,web)
    api=api or public_client()
    from huggingface_hub.errors import RepositoryNotFoundError
    try:dataset=tree(api,"dataset")
    except RepositoryNotFoundError:dataset=None
    if dataset is not None and set(dataset["tree"])-{".gitattributes"}:raise PublicationError("Fixed new dataset already contains material.")
    space=tree(api,"space")
    if any(name.startswith(NAMESPACE) or name in ("action-capture",NAMESPACE.rstrip("/")) for name in space["tree"]):
        raise PublicationError("The append-only capture Space namespace is already occupied.")
    downloader=downloader or (lambda repo,name,kind,revision:download(repo,name,kind,revision,output.parent/'capture-stage-anonymous-cache'))
    attrs=downloader(REPOS['space'],'.gitattributes','space',space['commit'])
    compare({'.gitattributes':space['tree']['.gitattributes']},{'.gitattributes':{
        'bytes':len(attrs),'blob':sha1(b'blob '+str(len(attrs)).encode()+b'\0'+attrs).hexdigest(),'lfs_sha256':None}})
    if b'*.mp4 filter=lfs diff=lfs merge=lfs -text' in attrs.splitlines():
        raise PublicationError('Small MP4 Git projection differs from the existing Space attribute mapping.')
    if tree(api,"space")!=space:raise PublicationError("Space moved while capturing its complete baseline.")
    output.mkdir(parents=True)
    for kind,payload in (("dataset",dataset_files),("space",space_files)):
        for name,raw in payload.items():
            relative(name);scan(raw);target=output/kind/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    rows=[{"kind":kind,"repo":REPOS[kind],"root":str(output/kind),"files":files(output/kind),"baseline":dataset if kind=="dataset" else space}for kind in REPOS]
    if any(item["path"] in space["tree"] for item in rows[1]["files"]):raise PublicationError("Capture namespace would overwrite an existing Space path.")
    plan={"schema":SCHEMA,"public":True,"namespace":NAMESPACE,"repositories":rows,"implementation":implementation(),
          **evidence,"qualified_robot_skill":False,"physical_robot_ready":False}
    atomic(output/"plan.json",plan,exclusive=True);checked(output/"plan.json");return plan


def checked(path):
    path = plain(path); plan = load(path)
    if (plan.get("schema") != SCHEMA or plan.get("public") is not True or plan.get("namespace") != NAMESPACE
            or plan.get("qualified_robot_skill") is not False or plan.get("physical_robot_ready") is not False
            or plan.get("implementation") != implementation()
            or [row.get("kind") for row in plan.get("repositories", [])] != list(REPOS)):
        raise PublicationError("Plan differs from the fixed reviewed Capture publication contract.")
    for row in plan["repositories"]:
        kind = row["kind"]
        if row["repo"] != REPOS[kind] or plain(row["root"]) != path.parent/kind or files(row["root"]) != row["files"]:
            raise PublicationError("Publication destination, snapshot or bytes changed.")
        if kind == "space" and (row["baseline"] is None or any(not item["path"].startswith(NAMESPACE) for item in row["files"])):
            raise PublicationError("Space writes must remain inside the new namespace.")
        baseline = row["baseline"]
        if baseline is not None and (not OID.fullmatch(baseline.get("commit", "")) or not isinstance(baseline.get("tree"), dict)):
            raise PublicationError("Staged baseline lacks an exact commit and tree.")
        if kind == "space" and any(name.startswith(NAMESPACE) or name in ("action-capture", NAMESPACE.rstrip("/")) for name in baseline["tree"]):
            raise PublicationError("Append-only namespace must be absent from the reviewed baseline.")
        if kind == "dataset" and baseline is not None and set(baseline["tree"])-{".gitattributes"}:
            raise PublicationError("The reviewed dataset baseline cannot contain unrelated material.")
    return plan


def _publish(path, api, token, operation_cls, resume_unchanged_parent=False):
    path = plain(path); plan = checked(path); plan_id = digest(read(path, MAX_PLAN))
    journal = path.parent/"journal.json"
    state = load(journal) if journal.exists() else {"schema": SCHEMA, "plan": plan_id, "repositories": {}}
    if state.get("schema") != SCHEMA or state.get("plan") != plan_id:
        raise PublicationError("Existing durable intent belongs to another plan.")
    if "dvidia" not in {str(org.get("name", "")).lower() for org in api.whoami().get("orgs", [])}:
        raise PublicationError("The publisher is not a member of the fixed Dvidia organization.")
    for row in plan['repositories']:
        if row['kind'] not in state['repositories'] and row['baseline'] is not None and tree(api,row['kind'])!=row['baseline']:
            raise PublicationError('Destination moved before the first remote mutation.')
    for row in plan["repositories"]:
        kind = row["kind"]; value = state["repositories"].get(kind)
        if value is None:
            baseline = row["baseline"]
            if baseline is None:
                state["repositories"][kind] = {"creating": True}; atomic(journal, state)
                api.create_repo(REPOS[kind], repo_type=kind, private=False, exist_ok=False)
                baseline = tree(api, kind)
                if set(baseline["tree"])-{".gitattributes"}:
                    raise PublicationError("New dataset contains an unexpected tree.")
            value = {"parent": baseline["commit"], "baseline": baseline["tree"], "completed": [], "pending": None}
            state["repositories"][kind] = value; atomic(journal, state)
        elif value.get("creating"):
            baseline = tree(api, kind)
            if kind != "dataset" or set(baseline["tree"])-{".gitattributes"}:
                raise PublicationError("Uncertain repository creation produced an unexpected destination.")
            value = {"parent": baseline["commit"], "baseline": baseline["tree"], "completed": [], "pending": None}
            state["repositories"][kind] = value; atomic(journal, state)
        batches = [row["files"][start:start+100] for start in range(0, len(row["files"]), 100)]
        if (row["baseline"] is not None and (value["parent"] != row["baseline"]["commit"] or value["baseline"] != row["baseline"]["tree"])
                or len(value["completed"]) > len(batches) or any(not OID.fullmatch(oid or "") for oid in value["completed"])):
            raise PublicationError("Durable journal differs from the reviewed parent or completion roster.")
        for index, batch in enumerate(batches):
            cumulative = sum(len(items) for items in batches[:index+1])
            wanted = expected_tree(row, value["baseline"], cumulative)
            if index < len(value["completed"]):
                continue
            parent = value["completed"][-1] if value["completed"] else value["parent"]
            title = "DVIDIA PiPER action capture v01 "+plan_id["sha256"]+" "+kind+" "+str(index)
            if value["pending"] is not None and value["pending"] != {"index": index, "parent": parent, "title": title}:
                raise PublicationError("Pending intent differs from the exact next reviewed commit.")
            head = tree(api, kind)
            if value["pending"] and head["commit"] == parent and not resume_unchanged_parent:
                raise PublicationError("Pending reply retains the unchanged parent; explicitly review before a guarded resume.")
            if value["pending"] and head["commit"] != parent:
                history = api.list_repo_commits(REPOS[kind], repo_type=kind, revision=head["commit"], token=False)
                if (len(history) < 2 or history[0].commit_id != head["commit"] or history[0].title != title
                        or history[1].commit_id != parent):
                    raise PublicationError("Uncertain write is not the exact intended immediate child.")
                compare(head["tree"], wanted)
                value["completed"].append(head["commit"]); value["pending"] = None; atomic(journal, state)
                continue
            prior = expected_tree(row, value["baseline"], cumulative-len(batch))
            if head["commit"] != parent:
                raise PublicationError("Destination moved before its parent-CAS write.")
            compare(head["tree"], prior)
            operations = []
            for item in batch:
                raw = read(Path(row["root"])/relative(item["path"]))
                if digest(raw) != {key: item[key] for key in ("bytes", "sha256")} or token.encode() in raw:
                    raise PublicationError("Upload bytes changed or contain the publishing credential.")
                operation = operation_cls(path_in_repo=item["path"], path_or_fileobj=raw)
                operation._upload_mode = "lfs" if item["representation"] == "lfs" else "regular"
                operation._should_ignore = False
                operations.append(operation)
            value["pending"] = {"index": index, "parent": parent, "title": title}; atomic(journal, state)
            try:
                committed = api.create_commit(REPOS[kind], repo_type=kind, parent_commit=parent,
                    operations=operations, commit_message=title)
            except Exception:
                raise PublicationError("Commit reply uncertain; preserve the pending journal for reconciliation.") from None
            head = tree(api, kind)
            if head["commit"] != committed.oid:
                raise PublicationError("Repository moved after the guarded commit.")
            compare(head["tree"], wanted); checked(path)
            value["completed"].append(head["commit"]); value["pending"] = None; atomic(journal, state)
            print(json.dumps({"kind": kind, "batch": index, "commit": head["commit"], "files": len(batch)}), flush=True)
        final = tree(api, kind)
        if final["commit"] != value["completed"][-1]:
            raise PublicationError("Completed repository moved.")
        compare(final["tree"], expected_tree(row, value["baseline"]))
    receipt = {"schema": SCHEMA, "plan": plan_id, "repositories": state["repositories"],
               "all_git_lfs_identities_verified": True, "anonymous_full_readback": False, "physical_robot_ready": False}
    atomic(path.parent/"publication-receipt.json", receipt)
    return receipt


@contextmanager
def _lock(folder):
    import fcntl
    descriptor=os.open(plain(folder)/'publisher.lock',os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise PublicationError('Another capture publisher is active.') from None
        yield
    finally:os.close(descriptor)


def publish(path,*,resume_unchanged_parent=False):
    path = plain(path)
    with _lock(path.parent):
        checked(path)
        api, token = client(path.parent/"publisher-cache")
        from huggingface_hub import CommitOperationAdd
        return _publish(path, api, token, CommitOperationAdd,resume_unchanged_parent=resume_unchanged_parent)


def verify(path, *, api=None, downloader=None, force_download=False):
    path = plain(path); plan = checked(path); receipt = load(path.parent/"publication-receipt.json")
    journal=load(path.parent/'journal.json')
    if (receipt.get('schema')!=SCHEMA or receipt["plan"] != digest(read(path, MAX_PLAN))
            or journal.get('schema')!=SCHEMA or journal.get('plan')!=receipt['plan']
            or receipt.get('repositories')!=journal.get('repositories') or set(receipt['repositories'])!=set(REPOS)):
        raise PublicationError("Completed receipt belongs to another plan.")
    api = api or public_client(); proofs = []
    for row in plan["repositories"]:
        kind = row["kind"]; value = receipt["repositories"][kind]; revision = value["completed"][-1]
        if (value.get('pending') is not None or len(value['completed'])!=(len(row['files'])+99)//100
                or any(not OID.fullmatch(oid or '') for oid in value['completed'])
                or row['baseline'] is not None and (value['parent']!=row['baseline']['commit'] or value['baseline']!=row['baseline']['tree'])
                or row['baseline'] is None and (kind!='dataset' or set(value['baseline'])-{'.gitattributes'})):
            raise PublicationError('Completed readback state differs from the reviewed exact baseline and full batch roster.')
        head = tree(api, kind)
        if head["commit"] != revision:
            raise PublicationError("Repository moved before anonymous readback.")
        compare(head["tree"], expected_tree(row, value["baseline"]))
        def one(item):
            raw = (downloader(REPOS[kind], item["path"], kind, revision) if downloader else
                   download(REPOS[kind], item["path"], kind, revision, path.parent/"anonymous-readback-cache",force_download=force_download))
            if digest(raw) != {key: item[key] for key in ("bytes", "sha256")}:
                raise PublicationError("Anonymous pinned download differs from its staged bytes.")
            return {"path": item["path"], **digest(raw)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            downloaded = list(pool.map(one, row["files"]))
        if tree(api, kind)["commit"] != revision:
            raise PublicationError("Repository moved during anonymous readback.")
        proofs.append({"kind": kind, "repo": REPOS[kind], "commit": revision, "files": downloaded})
    result = {"schema": SCHEMA, "plan": receipt["plan"], "repositories": proofs,
              "anonymous_full_readback": True, "force_download":force_download,"physical_robot_ready": False}
    atomic(path.parent/"anonymous-readback.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "check", "publish", "verify"))
    parser.add_argument("--kit"); parser.add_argument("--web"); parser.add_argument("--output"); parser.add_argument("--plan")
    parser.add_argument('--resume-unchanged-parent',action='store_true')
    parser.add_argument('--force-download',action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == "stage":
            result = stage(args.kit, args.web, args.output)
        elif args.command == "check":
            result = checked(args.plan)
        elif args.command == "publish":
            result = publish(args.plan,resume_unchanged_parent=args.resume_unchanged_parent)
        else:
            result = verify(args.plan,force_download=args.force_download)
        print(json.dumps({"schema": result["schema"], "command": args.command, "completed": True}), flush=True)
        return 0
    except Exception as exc:
        print("Capture publication stopped: "+(str(exc) if isinstance(exc, PublicationError) else type(exc).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
