"""Offline appearance adapter for NVIDIA PhysicalAI-SimReady-Materials v0.2.0.

This module deliberately handles only the verified Fabric category archive. It
extracts a single declared base-color JPEG, resizes it with FFmpeg, and adds it
as a MuJoCo RGB texture. It does not evaluate USD, MaterialX, or OpenPBR graphs
and never changes physics parameters.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import zlib
import zipfile


REPOSITORY = "NVIDIA-Omniverse/PhysicalAI-SimReady-Materials"
RELEASE = "v0.2.0"
RELEASE_URL = f"https://github.com/{REPOSITORY}/releases/tag/{RELEASE}"
SOURCE_SHA = "8f80fbd"
LICENSE = "MIT-0"
ARCHIVE_SIZE = 45_574_867
ARCHIVE_SHA256 = "a68ca7376ed4abd6160ff7fed234cd1012bded73fd17f1588adcd34a91fe7d42"
SCHEMA = "dvidia.material-appearance/v1"

_MAX_MEMBERS = 256
_MAX_TOTAL_SIZE = 64 * 1024 * 1024
_MAX_MEMBER_SIZE = 8 * 1024 * 1024
_MAX_RATIO = 200
_MATERIAL_RE = re.compile(r"^Materials/Fabric/([A-Za-z0-9_-]+)\.usda$")
_DIFFUSE_RE = re.compile(r"(?m)^\s*asset inputs:base_color_texture_file\s*=\s*@([^@\r\n]+)@")
_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,47}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_LIMITATIONS = [
    "Only the declared base-color diffuse JPEG is used.",
    "Normal, roughness, weight, fuzz, and other OpenPBR/MaterialX channels are not evaluated.",
    "Upstream UV tiling and offset controls are not applied; MuJoCo geom UVs are used unchanged.",
    "This is an albedo appearance approximation, not a full USD material import.",
    "No mass, friction, restitution, stiffness, damping, or other physics values are inferred or changed.",
]


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _safe_zip_inventory(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Return a bounded, unambiguous ZIP inventory without extracting members."""
    infos = archive.infolist()
    if len(infos) > _MAX_MEMBERS:
        raise ValueError("Archive contains too many entries")
    result: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in infos:
        name = info.filename
        if not isinstance(name, str) or "\\" in name or "\x00" in name:
            raise ValueError("Archive contains an unsafe member name")
        clean = name[:-1] if name.endswith("/") else name
        parts = clean.split("/")
        if (PurePosixPath(clean).is_absolute() or not clean or
                any(part in ("", ".", "..") for part in parts)):
            raise ValueError("Archive contains an unsafe member path")
        if name in result:
            raise ValueError("Archive contains duplicate member names")
        if info.flag_bits & 0x1:
            raise ValueError("Encrypted archive members are not supported")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError("Archive symlinks are not supported")
        if info.file_size < 0 or info.file_size > _MAX_MEMBER_SIZE:
            raise ValueError("Archive member exceeds the size limit")
        total += info.file_size
        if total > _MAX_TOTAL_SIZE:
            raise ValueError("Archive exceeds the expanded size limit")
        if info.file_size and (info.compress_size == 0 or info.file_size / info.compress_size > _MAX_RATIO):
            raise ValueError("Archive member exceeds the compression-ratio limit")
        result[name] = info
    return result


def _verified_archive(path: str | os.PathLike[str]) -> tuple[Path, str, zipfile.ZipFile, dict[str, zipfile.ZipInfo]]:
    archive_path = Path(path)
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError("archive must be a local regular file")
    if archive_path.stat().st_size != ARCHIVE_SIZE:
        raise ValueError("archive size does not match NVIDIA Fabric v0.2.0")
    digest = _sha(archive_path.read_bytes())
    if digest != ARCHIVE_SHA256:
        raise ValueError("archive SHA-256 does not match NVIDIA Fabric v0.2.0")
    archive = zipfile.ZipFile(archive_path, "r")
    try:
        inventory = _safe_zip_inventory(archive)
        return archive_path, digest, archive, inventory
    except Exception:
        archive.close()
        raise


def _catalog(archive: zipfile.ZipFile, digest: str,
             inventory: dict[str, zipfile.ZipInfo]) -> dict:
    materials = []
    for member in sorted(name for name in inventory if _MATERIAL_RE.fullmatch(name)):
        match = _MATERIAL_RE.fullmatch(member)
        assert match is not None
        material_id = match.group(1)
        source = archive.read(member)
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Material {material_id} is not UTF-8 USDA") from exc
        texture_matches = _DIFFUSE_RE.findall(text)
        if len(texture_matches) != 1:
            raise ValueError(f"Material {material_id} must declare one base-color texture")
        relative = texture_matches[0]
        texture_path = PurePosixPath(relative)
        if (texture_path.is_absolute() or "\\" in relative or
                any(part in ("", ".", "..") for part in texture_path.parts) or
                texture_path.suffix.lower() not in (".jpg", ".jpeg")):
            raise ValueError(f"Material {material_id} has an unsupported diffuse texture path")
        package_path = f"Materials/Fabric/{relative}"
        if package_path not in inventory:
            raise ValueError(f"Material {material_id} references a missing texture")
        texture = archive.read(package_path)
        materials.append({
            "id": material_id,
            "material_path": member,
            "material_sha256": _sha(source),
            "diffuse_texture_path": package_path,
            "diffuse_texture_sha256": _sha(texture),
            "diffuse_texture_bytes": len(texture),
        })
    if not materials:
        raise ValueError("Archive contains no supported Fabric materials")
    return {
        "repository": REPOSITORY,
        "release": RELEASE,
        "release_url": RELEASE_URL,
        "source_sha": SOURCE_SHA,
        "license": LICENSE,
        "archive_sha256": digest,
        "materials": materials,
        "limitations": list(_LIMITATIONS),
    }


def catalog_archive(path: str | os.PathLike[str]) -> dict:
    """Verify and list the known Fabric v0.2.0 archive without extracting it."""
    _, digest, archive, inventory = _verified_archive(path)
    try:
        return _catalog(archive, digest, inventory)
    finally:
        archive.close()


def _decode_diffuse(jpeg: bytes, size: int) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to decode the diffuse JPEG")
    command = [ffmpeg, "-v", "error", "-nostdin", "-i", "pipe:0", "-frames:v", "1",
               "-vf", f"scale={size}:{size}:flags=bilinear", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "pipe:1"]
    try:
        result = subprocess.run(command, input=jpeg, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True, timeout=20)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("FFmpeg could not decode the selected diffuse JPEG") from exc
    pixels = size * size * 3
    if len(result.stdout) != pixels:
        raise ValueError("FFmpeg returned an unexpected RGB24 byte count")
    return result.stdout


def _png_rgb(rgb: bytes, size: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xffffffff)

    rows = b"".join(b"\x00" + rgb[y * size * 3:(y + 1) * size * 3] for y in range(size))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(rows, level=6)) + chunk(b"IEND", b""))


def prepare_appearance(archive_path: str | os.PathLike[str], material_id: str,
                       output: str | os.PathLike[str], texture_size: int = 256) -> dict:
    """Prepare one selected diffuse image and strict provenance manifest offline."""
    if not isinstance(material_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", material_id):
        raise ValueError("material_id must be a catalog material ID")
    if isinstance(texture_size, bool) or not isinstance(texture_size, int) or not 64 <= texture_size <= 512:
        raise ValueError("texture_size must be an integer from 64 through 512")
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise ValueError("output must be a new directory path")
    _, digest, archive, inventory = _verified_archive(archive_path)
    try:
        catalog = _catalog(archive, digest, inventory)
        selected = next((row for row in catalog["materials"] if row["id"] == material_id), None)
        if selected is None:
            raise ValueError("material_id is not in the verified Fabric archive")
        diffuse = archive.read(selected["diffuse_texture_path"])
    finally:
        archive.close()
    rgb = _decode_diffuse(diffuse, texture_size)
    png = _png_rgb(rgb, texture_size)
    target.mkdir(parents=True, exist_ok=False)
    try:
        (target / "texture.rgb").write_bytes(rgb)
        (target / "texture.png").write_bytes(png)
        manifest = {
            "schema": SCHEMA,
            "material_id": material_id,
            "source": {
                "repository": REPOSITORY, "release": RELEASE, "release_url": RELEASE_URL,
                "source_sha": SOURCE_SHA, "license": LICENSE,
                "archive_sha256": digest,
                "material_path": selected["material_path"],
                "material_sha256": selected["material_sha256"],
                "diffuse_texture_path": selected["diffuse_texture_path"],
                "diffuse_texture_sha256": selected["diffuse_texture_sha256"],
                "diffuse_texture_bytes": selected["diffuse_texture_bytes"],
            },
            "texture": {
                "width": texture_size, "height": texture_size, "channels": 3,
                "pixel_format": "RGB24", "resize_method": "FFmpeg bilinear exact-square resize",
                "rgb_path": "texture.rgb", "rgb_bytes": len(rgb), "rgb_sha256": _sha(rgb),
                "png_path": "texture.png", "png_bytes": len(png), "png_sha256": _sha(png),
            },
            "appearance_kind": "diffuse_albedo_approximation",
            "limitations": catalog["limitations"],
        }
        (target / "appearance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _strict_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("appearance manifest is missing or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("appearance manifest must be an object")
    return value


def load_appearance(directory: str | os.PathLike[str]):
    """Validate a prepared appearance and return its metadata and uint8 RGB array."""
    import numpy as np

    folder = Path(directory)
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("appearance must be a local directory")
    manifest = _strict_json(folder / "appearance.json")
    if set(manifest) != {"schema", "material_id", "source", "texture", "appearance_kind", "limitations"}:
        raise ValueError("appearance manifest has missing or unknown fields")
    if manifest["schema"] != SCHEMA or manifest["appearance_kind"] != "diffuse_albedo_approximation":
        raise ValueError("unsupported appearance manifest schema or kind")
    if not isinstance(manifest["material_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", manifest["material_id"]):
        raise ValueError("appearance material ID is invalid")
    source = manifest["source"]
    expected_source_keys = {"repository", "release", "release_url", "source_sha", "license", "archive_sha256",
                            "material_path", "material_sha256", "diffuse_texture_path", "diffuse_texture_sha256",
                            "diffuse_texture_bytes"}
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise ValueError("appearance source provenance is invalid")
    if (source["repository"], source["release"], source["release_url"], source["source_sha"], source["license"]) != (
            REPOSITORY, RELEASE, RELEASE_URL, SOURCE_SHA, LICENSE):
        raise ValueError("appearance source is not the pinned NVIDIA Fabric release")
    if source["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("appearance archive hash is not the pinned Fabric archive")
    for field in ("archive_sha256", "material_sha256", "diffuse_texture_sha256"):
        if not isinstance(source[field], str) or not _SHA_RE.fullmatch(source[field]):
            raise ValueError("appearance source hash is invalid")
    match = _MATERIAL_RE.fullmatch(source["material_path"] if isinstance(source["material_path"], str) else "")
    if not match or match.group(1) != manifest["material_id"]:
        raise ValueError("appearance material path does not match its ID")
    expected_diffuse_path = f"Materials/Fabric/textures/{manifest['material_id']}_diffuse.jpg"
    if source["diffuse_texture_path"] != expected_diffuse_path:
        raise ValueError("appearance diffuse source path is invalid")
    if type(source["diffuse_texture_bytes"]) is not int or not 0 < source["diffuse_texture_bytes"] <= _MAX_MEMBER_SIZE:
        raise ValueError("appearance diffuse source size is invalid")
    texture = manifest["texture"]
    expected_texture_keys = {"width", "height", "channels", "pixel_format", "resize_method", "rgb_path", "rgb_bytes", "rgb_sha256", "png_path", "png_bytes", "png_sha256"}
    if not isinstance(texture, dict) or set(texture) != expected_texture_keys:
        raise ValueError("appearance texture metadata is invalid")
    width, height = texture["width"], texture["height"]
    if (type(width) is not int or type(height) is not int or width != height or not 64 <= width <= 512 or
            texture["channels"] != 3 or texture["pixel_format"] != "RGB24" or
            texture["resize_method"] != "FFmpeg bilinear exact-square resize"):
        raise ValueError("appearance texture dimensions or encoding are unsupported")
    if texture["rgb_path"] != "texture.rgb" or texture["png_path"] != "texture.png":
        raise ValueError("appearance texture path is invalid")
    for field in ("rgb_sha256", "png_sha256"):
        if not isinstance(texture[field], str) or not _SHA_RE.fullmatch(texture[field]):
            raise ValueError("appearance texture hash is invalid")
    expected_bytes = width * height * 3
    if texture["rgb_bytes"] != expected_bytes or type(texture["rgb_bytes"]) is not int:
        raise ValueError("appearance RGB byte count does not match its dimensions")
    if type(texture["png_bytes"]) is not int or not 0 < texture["png_bytes"] <= 1_000_000:
        raise ValueError("appearance PNG byte count exceeds its limit")
    rgb_path, png_path = folder / "texture.rgb", folder / "texture.png"
    if rgb_path.is_symlink() or png_path.is_symlink() or not rgb_path.is_file() or not png_path.is_file():
        raise ValueError("appearance texture files are missing or unsafe")
    rgb, png = rgb_path.read_bytes(), png_path.read_bytes()
    if len(rgb) != expected_bytes or _sha(rgb) != texture["rgb_sha256"]:
        raise ValueError("appearance RGB data failed its size or hash check")
    if len(png) != texture["png_bytes"] or _sha(png) != texture["png_sha256"] or not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("appearance PNG failed its size or hash check")
    if manifest["limitations"] != _LIMITATIONS:
        raise ValueError("appearance limitations differ from the adapter contract")
    return manifest, np.frombuffer(rgb, dtype=np.uint8).reshape((height, width, 3)).copy()


def apply_to_spec(spec, directory: str | os.PathLike[str], geom_names,
                  prefix: str = "dvidia_fabric") -> dict:
    """Bind the appearance texture to selected geoms without changing mechanics."""
    import mujoco as mj

    metadata, rgb = load_appearance(directory)
    if not isinstance(geom_names, (list, tuple)) or not geom_names or any(
            not isinstance(name, str) or not name for name in geom_names):
        raise ValueError("geom_names must be a nonempty list or tuple of names")
    if len(set(geom_names)) != len(geom_names):
        raise ValueError("geom_names must not contain duplicates")
    if not isinstance(prefix, str) or not _PREFIX_RE.fullmatch(prefix):
        raise ValueError("prefix must be a short MuJoCo-safe name")
    existing = {geom.name for geom in spec.geoms}
    if any(name not in existing for name in geom_names):
        raise ValueError("geom_names contains a geom missing from the MuJoCo spec")
    texture_name = prefix + "_texture"
    material_name = prefix + "_material"
    if texture_name in {item.name for item in spec.textures} or material_name in {item.name for item in spec.materials}:
        raise ValueError("appearance prefix collides with an existing texture or material")
    size = int(rgb.shape[0])
    texture = spec.add_texture(name=texture_name, type=mj.mjtTexture.mjTEXTURE_2D,
                               width=size, height=size, nchannel=3)
    texture.data = rgb.tobytes()
    material = spec.add_material(name=material_name)
    material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = texture_name
    material.rgba = [1, 1, 1, 1]
    for geom in spec.geoms:
        if geom.name in geom_names:
            geom.material = material_name
            geom.rgba = [1, 1, 1, 1]
    return {"material_id": metadata["material_id"], "geom_names": list(geom_names),
            "texture_name": texture_name, "material_name": material_name,
            "archive_sha256": metadata["source"]["archive_sha256"],
            "material_sha256": metadata["source"]["material_sha256"],
            "diffuse_texture_sha256": metadata["source"]["diffuse_texture_sha256"]}


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline NVIDIA v0.2.0 Fabric appearance adapter")
    sub = parser.add_subparsers(dest="command", required=True)
    catalog_parser = sub.add_parser("catalog", help="verify and list a local Fabric archive")
    catalog_parser.add_argument("archive")
    prepare_parser = sub.add_parser("prepare", help="prepare one Fabric diffuse appearance")
    prepare_parser.add_argument("archive")
    prepare_parser.add_argument("--material", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--size", type=int, default=256)
    inspect_parser = sub.add_parser("inspect", help="validate and summarize a prepared appearance")
    inspect_parser.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "catalog":
        result = catalog_archive(args.archive)
    elif args.command == "prepare":
        result = prepare_appearance(args.archive, args.material, args.output, args.size)
    else:
        result, _ = load_appearance(args.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    """Command-line entry point for ``python -m dvidia_training.material_appearance``."""
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(_main())
