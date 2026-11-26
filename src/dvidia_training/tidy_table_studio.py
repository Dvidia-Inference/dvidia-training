"""Local, shared records for the Skillspace table UI and Terminal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from .camera_action_studio import _local_path

DEFAULT_DIRECTORY = Path("runs/training-studio")
DEFAULT_ASSETS = Path("robot-assets")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _latest(root, kind):
    pointer = root / f"latest-{kind}.json"
    if not pointer.exists():
        return None
    key = json.loads(pointer.read_text())["id"]
    if not isinstance(key, str) or uuid.UUID(hex=key).hex != key:
        raise ValueError("Invalid table record identifier.")
    record = json.loads((root / kind / (key + ".json")).read_text())
    if record.get("id") != key:
        raise ValueError("The table record does not match its identifier.")
    return record


def table_state(directory):
    root = Path(directory) / "tables"
    reference = _latest(root, "references")
    run = _latest(root, "runs")
    if not reference or (run and run["reference_id"] != reference["id"]):
        run = None
    return {"reference": reference, "run": run}


def _validate_layout(layout, *, reference=False):
    from .tidy_table import OBJECTS, ZONES
    if (type(layout) is not dict or set(layout) != set(OBJECTS)
            or any(type(zone) is not str or zone not in ZONES for zone in layout.values())
            or len(set(layout.values())) != len(OBJECTS)):
        raise ValueError("Place each of the three blocks in a different zone.")
    if reference and "buffer" in layout.values():
        raise ValueError("Leave the buffer free when remembering a tidy table.")
    return {obj: layout[obj] for obj in OBJECTS}


def table_worker(kind, directory, assets, layout, allow_exchange, expected_reference=None):
    from .tidy_table import capture_table, run_table_case
    layout = _validate_layout(layout, reference=kind == "remember")
    assets = _local_path(str(assets), "PiPER assets")
    root = Path(directory) / "tables"
    identity = uuid.uuid4().hex
    first_frame = last_frame = None

    def receive_frame(rgb):
        nonlocal first_frame, last_frame
        last_frame = rgb.copy()
        if first_frame is None:
            first_frame = last_frame

    if kind == "remember":
        if type(allow_exchange) is not bool:
            raise ValueError("Choose whether blocks A and B may trade places.")
        capture = capture_table(layout, assets=assets, interchangeable=[("A", "B")] if allow_exchange else [],
                                frame_sink=receive_frame)
        if not isinstance(capture.get("reference"), dict):
            raise ValueError("PiPER could not observe a complete reference.")
        key = "references"
        record = {"schema": "dvidia.remembered-table/v1", "id": identity,
                  "layout": layout, "allow_exchange": allow_exchange, "capture": capture}
    elif kind == "tidy":
        saved = _latest(root, "references")
        if not saved or (expected_reference is not None and expected_reference != saved["id"]):
            raise ValueError("Load the current remembered table before tidying.")
        result = run_table_case(saved["capture"]["reference"], layout, assets=assets, frame_sink=receive_frame)
        key = "runs"
        record = {"schema": "dvidia.table-studio-run/v1", "id": identity,
                  "reference_id": saved["id"], "layout": layout, "result": result}
    else:
        raise ValueError("Unknown table action.")
    if last_frame is not None:
        from .piper_action_capture import _png
        frames = root / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        with (frames / (identity + ".png")).open("xb") as image:
            image.write(_png(last_frame))
        record["frame"] = "frames/" + identity + ".png"
        if kind == "tidy":
            record["first_frame"] = record["frame"]
            if first_frame is not last_frame:
                with (frames / (identity + "-start.png")).open("xb") as image:
                    image.write(_png(first_frame))
                record["first_frame"] = "frames/" + identity + "-start.png"
    _write(root / key / (identity + ".json"), record)
    _write(root / f"latest-{key}.json", {"id": identity})
    return record


class TableStudio:
    def __init__(self, server, assets):
        self.server = server
        self.assets = str(Path(assets).expanduser().absolute())
        self.future = None
        self.kind = None

    def busy(self):
        return self.future is not None and not self.future.done()

    def state(self):
        with self.server.lock:
            value = table_state(self.server.directory)
            error = None
            if self.future is not None and self.future.done():
                failure = self.future.exception()
                error = str(failure)[:1500] if failure is not None else None
            self.server.refresh()
            other_busy = self.server.robot.busy() or bool(self.server.job and self.server.job["status"] == "running")
            return {**value, "busy": self.busy(), "other_busy": other_busy,
                    "operation": self.kind, "error": error, "assets": self.assets}

    def submit(self, kind, body):
        expected = {"layout", "assets", "allow_exchange" if kind == "remember" else "reference_id"}
        if type(body) is not dict or set(body) != expected:
            raise ValueError("The table request is incomplete.")
        layout = _validate_layout(body["layout"], reference=kind == "remember")
        assets = _local_path(body["assets"], "PiPER assets")
        if kind == "remember" and type(body["allow_exchange"]) is not bool:
            raise ValueError("Choose whether blocks A and B may trade places.")
        with self.server.lock:
            state = self.state()
            if state["busy"] or state["other_busy"]:
                raise RuntimeError("One CPU job is already running. Wait for its outcome.")
            if kind == "tidy" and (not state["reference"] or body["reference_id"] != state["reference"]["id"]):
                raise ValueError("Load the current remembered table before tidying.")
            self.future = self.server.pool.submit(table_worker, kind, self.server.directory, assets,
                                                  layout, body.get("allow_exchange", False), body.get("reference_id"))
            self.kind = kind
            self.assets = str(assets)
        return {"operation": kind, "status": "running"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Remember a simulated table and tidy it with PiPER.")
    actions = parser.add_subparsers(dest="command", required=True)
    for kind in ("remember", "tidy", "inspect"):
        command = actions.add_parser(kind)
        command.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY, help="Shared Training Studio directory.")
        if kind != "inspect":
            command.add_argument("--assets", type=Path, default=DEFAULT_ASSETS, help="Verified local PiPER assets.")
            command.add_argument("--layout", default="left,center,right", help="Positions of blocks A,B,C in order: left, center, right or buffer.")
        if kind == "remember":
            command.add_argument("--exchange-ab", action="store_true", help="Accept A and B in each other's home zones.")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            record = table_state(args.directory)
        else:
            from .tidy_table import OBJECTS
            parts = [part.strip() for part in args.layout.split(",")]
            if len(parts) != len(OBJECTS):
                raise ValueError("Use three comma-separated zones in A,B,C order.")
            record = table_worker(args.command, args.directory, args.assets, dict(zip(OBJECTS, parts)),
                                  getattr(args, "exchange_ab", False))
        print(json.dumps(record, allow_nan=False, indent=2))
        if args.command == "tidy" and not record["result"].get("success"):
            return 1
        return 0
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
