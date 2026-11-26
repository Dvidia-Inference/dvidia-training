"""Provenance and offline resource loading for the standalone Unitree R1-7A."""
from hashlib import sha256
from importlib.resources import files

_REPO = "https://github.com/unitreerobotics/r1-7a_sdk"
_REVISION = "92f32514c6a9089c580ddf331eeb26075fcc5609"
_ARCHIVE_PATH = "urdf/URDF_R1-7A.tar.gz"
_ARCHIVE_SHA256 = "dca2cea68150891b5312108dd7c448103d0f0d67690fde0918cea23b41b65e25"
_LICENSE_SHA256 = "41c9cedfb396ab780982b80c5de1d4c0bcfa947fe93cee2265ddc0f8fb049709"
_DESCRIPTIONS = {
    "right": (
        "R1_Robotic_Arm_R_V1.2-1023/R1_Robotic_Arm_V1.2/R1_Robotic_Arm/urdf/R1_Robotic_Arm.urdf",
        "4e6655f511a88447e70a9437355241918b4ec094e4d8c09f1cae754d7724c0f5",
    ),
    "left": (
        "R1_Robotic_Arm_L_V1.0-1027/R1_Robotic_Arm_L_V1.0-1027/urdf/R1_Robotic_Arm_L_V1.0-1027.urdf",
        "e89fde206857e5740dad7980954679ead95e7e10516f466eb27277af5f8662f5",
    ),
}


def source_info(side: str) -> dict:
    """Identify the exact vendor description, source archive and license."""
    if side not in ("right", "left"):
        raise ValueError("Choose the Unitree R1-7A arm side: right or left.")
    member, digest = _DESCRIPTIONS[side]
    return {
        "side": side,
        "repository": _REPO,
        "revision": _REVISION,
        "archive_url": f"{_REPO}/raw/{_REVISION}/{_ARCHIVE_PATH}",
        "archive_sha256": _ARCHIVE_SHA256,
        "archive_member": member,
        "urdf_sha256": digest,
        "license": "BSD-3-Clause",
        "license_url": f"{_REPO}/blob/{_REVISION}/LICENSE",
        "license_sha256": _LICENSE_SHA256,
    }


def urdf_bytes(side: str) -> bytes:
    """Return the side-specific, original URDF after checking its digest."""
    info = source_info(side)
    resource = files("dvidia_training").joinpath("unitree_r1_7a", f"{side}.urdf")
    data = resource.read_bytes()
    if sha256(data).hexdigest() != info["urdf_sha256"]:
        raise ValueError(f"Unitree R1-7A {side} URDF differs from the pinned vendor bytes.")
    return data
