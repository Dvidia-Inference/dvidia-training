"""Build both static bare-arm viewer meshes from Unitree's pinned SDK tarball.

Use PYTHONPATH=src python scripts/build-unitree-r1-viewer-assets.py ARCHIVE OUTPUT.
The little-endian vertex positions are in link visual coordinates, metres; the
Uint32 index stream preserves the vendor STL triangles and original winding.
"""
import argparse
from hashlib import sha256
from importlib.resources import files
import io
import json
from pathlib import Path, PurePosixPath
import struct
import tarfile
from xml.etree import ElementTree

import numpy as np

from dvidia_training.unitree_r1_assets import source_info, urdf_bytes

LINKS = ["base_link", *[f"Link{i}" for i in range(1, 8)]]
TRIANGLE = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])


def read_member(archive, path):
    member = archive.getmember(str(path))
    if not member.isfile() or not 0 < member.size < 4_000_000:
        raise ValueError(f"Unexpected archive member: {path}")
    return archive.extractfile(member).read()


def vector(text):
    values = [float(value) for value in text.split()]
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError(f"Expected three finite coordinates: {text}")
    return values


def origin(element):
    frame = element.find("origin")
    return {name: vector(frame.get(name)) for name in ("xyz", "rpy")}


def indexed_geometry(stl):
    if len(stl) < 84:
        raise ValueError("Truncated binary STL header")
    faces = struct.unpack_from("<I", stl, 80)[0]
    if not faces or len(stl) != 84 + faces * TRIANGLE.itemsize:
        raise ValueError("Invalid binary STL triangle count")
    triangles = np.frombuffer(stl, dtype=TRIANGLE, count=faces, offset=84)
    corners = triangles["corners"].reshape(-1, 3)
    if not np.isfinite(corners).all() or np.abs(corners).max() > 1.0:
        raise ValueError("Coordinates exceed the metre-sized arm link envelope")
    positions, inverse = np.unique(corners, axis=0, return_inverse=True)
    indices = inverse.reshape(-1).astype("<u4")
    if int(indices.max()) >= len(positions) or not np.array_equal(positions[indices], corners):
        raise ValueError("Indexed surface differs from vendor triangles")
    return positions.astype("<f4", copy=False), indices


def build_side(archive, side):
    info = source_info(side)
    description = PurePosixPath(info["archive_member"])
    source = read_member(archive, description)
    if source != urdf_bytes(side):
        raise ValueError("Archived URDF differs from the packaged, pinned description")
    tree = ElementTree.fromstring(source)
    link_nodes = tree.findall("link")
    if [node.get("name") for node in link_nodes] != LINKS:
        raise ValueError("Unexpected vendor link chain")
    links, joints, buffer = [], [], bytearray()
    for node in link_nodes:
        name = node.get("name")
        visual = node.find("visual")
        mesh = visual.find("geometry/mesh")
        mesh_name = name + ".STL"
        if side == "left":
            path = description.parent.parent / "meshes" / mesh_name
            expected_uri = f"package://{description.parent.parent.name}/meshes/{mesh_name}"
        else:
            path = description.parent / mesh_name
            expected_uri = mesh_name
        if mesh.get("filename") != expected_uri:
            raise ValueError(f"Unexpected upstream mesh URI for {name}")
        scale = vector(mesh.get("scale", "1 1 1"))
        if scale != [1., 1., 1.]:
            raise ValueError("Expected native metre coordinates without scaling")
        positions, indices = indexed_geometry(read_member(archive, path))
        position_offset = len(buffer)
        buffer.extend(positions.tobytes())
        index_offset = len(buffer)
        buffer.extend(indices.tobytes())
        links.append(dict(name=name, **origin(visual), scale=scale,
                          position_byte_offset=position_offset, vertex_count=len(positions),
                          index_byte_offset=index_offset, index_count=len(indices)))
    for index, node in enumerate(tree.findall("joint"), start=1):
        name = node.get("name")
        parent, child = node.find("parent").get("link"), node.find("child").get("link")
        if index >= len(LINKS) or (name, parent, child, node.get("type")) != (
                f"J{index}", LINKS[index-1], LINKS[index], "revolute"):
            raise ValueError("Unexpected vendor joint chain")
        axis = vector(node.find("axis").get("xyz"))
        limit = node.find("limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if not np.isclose(np.linalg.norm(axis), 1) or not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"Invalid rotation axis or range for {name}")
        joints.append(dict(name=name, parent=parent, child=child, **origin(node),
                           axis=axis, lower=lower, upper=upper))
    if len(joints) != 7:
        raise ValueError("Expected seven revolute joints")
    record = dict(schema="dvidia.unitree-r1-7a-viewer/v1", side=side, binary=f"{side}.bin",
                  binary_sha256=sha256(buffer).hexdigest(), source=info, links=links, joints=joints)
    return record, buffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.archive.stat().st_size != 5_080_121:
        parser.error("Expected the pinned Unitree description archive")
    archive_bytes = args.archive.read_bytes()
    if sha256(archive_bytes).hexdigest() != source_info("right")["archive_sha256"]:
        parser.error("Unitree distribution checksum mismatch")
    outputs = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for side in ("left", "right"):
            record, binary = build_side(archive, side)
            outputs[f"{side}.bin"] = binary
            outputs[f"{side}.json"] = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    notice = files("dvidia_training").joinpath("unitree_r1_7a", "LICENSE.txt").read_bytes()
    if sha256(notice).hexdigest() != source_info("right")["license_sha256"]:
        parser.error("Unitree license checksum mismatch")
    outputs["UNITREE-LICENSE.txt"] = notice
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (args.output / name).write_bytes(data)


if __name__ == "__main__":
    main()
