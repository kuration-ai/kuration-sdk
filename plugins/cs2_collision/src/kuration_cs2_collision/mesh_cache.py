"""Map-geometry loading, ported from the reference `collision_classifier.py`
(`build_mesh_from_glb/bsp/nav`, `load_map_mesh`), plus an on-disk geometry cache layer
that the reference scripts did not have.

The reference scripts rebuild the full trimesh BVH from raw `.bsp`/`.glb`/`.nav` bytes on
every process that touches a given map -- expensive because of Source-engine lump-table
parsing and texture-name-based non-solid-face filtering (BSP), not because of the BVH
build itself. `load_mesh_cached()` adds a two-layer cache in front of the same parsers:
an in-memory dict (free reuse across every source a single pipeline run touches, since
Kuration's batch-transform stage runs single-process/sequential -- see the operator
module docstring) and an on-disk `.npz` of the already-parsed, already-filtered
(vertices, faces) arrays (reused across separate runs/processes, skipping the expensive
parse step entirely on a cache hit).
"""
from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np

COMPONENT_FMT = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}
COMPONENT_SIZE = {5121: 1, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

NON_SOLID_PATTERNS = ("skybox", "sky", "trigger", "clip", "hint", "invisible")
NON_SOLID_BSP = frozenset({
    "tools/toolsskybox", "tools/toolssky", "tools/toolsskybox2d",
    "tools/toolstrigger", "tools/toolsareaportal", "tools/toolsoccluder",
    "tools/toolshint", "tools/toolsskip", "tools/toolsinvisible",
    "tools/toolsblocklight", "tools/toolsblack",
})


def _trimesh():
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "kuration-plugin-cs2-collision needs trimesh: pip install trimesh"
        ) from e
    return trimesh


def _read_glb(glb_path: str):
    with open(glb_path, "rb") as f:
        magic, version, _ = struct.unpack("<4sII", f.read(12))
        assert magic == b"glTF" and version == 2, "Not a GLB 2.0 file"
        c0_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)
        gltf = json.loads(f.read(c0_len).decode("utf-8"))
        c1_hdr = f.read(8)
        bin_data = f.read(struct.unpack("<I", c1_hdr[:4])[0]) if len(c1_hdr) == 8 else b""
    return gltf, bin_data


def _read_accessor(gltf, bin_data, acc_idx):
    acc = gltf["accessors"][acc_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    n = acc["count"]
    fmt = COMPONENT_FMT[acc["componentType"]]
    sz = COMPONENT_SIZE[acc["componentType"]]
    nc = TYPE_COUNT[acc["type"]]
    stride = bv.get("byteStride", nc * sz)
    rows = [struct.unpack_from(f"{nc}{fmt}", bin_data, off + i * stride) for i in range(n)]
    return np.array(rows, dtype=np.float32)


def _build_geometry_from_glb(path: str) -> tuple:
    gltf, bin_data = _read_glb(path)
    all_verts, all_faces = [], []
    vert_offset = 0
    for mesh in gltf["meshes"]:
        name = mesh.get("name", "")
        if any(p in name.lower() for p in NON_SOLID_PATTERNS):
            continue
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
                continue
            verts = _read_accessor(gltf, bin_data, prim["attributes"]["POSITION"])
            if "indices" in prim:
                idx = _read_accessor(gltf, bin_data, prim["indices"]).astype(np.int32).flatten()
            else:
                idx = np.arange(len(verts), dtype=np.int32)
            faces = idx[: len(idx) // 3 * 3].reshape(-1, 3) + vert_offset
            all_verts.append(verts)
            all_faces.append(faces)
            vert_offset += len(verts)
    if not all_verts:
        raise ValueError(f"no solid geometry found in GLB: {path}")
    return np.concatenate(all_verts), np.concatenate(all_faces)


def _build_geometry_from_bsp(path: str) -> tuple:
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"VBSP", f"Not a VBSP file: {path}"

    def lump(idx):
        off_hdr = 8 + idx * 16
        ofs = struct.unpack_from("<i", data, off_hdr)[0]
        length = struct.unpack_from("<i", data, off_hdr + 4)[0]
        return ofs, length

    v_off, v_len = lump(3)
    n_verts = v_len // 12
    verts = np.frombuffer(data, dtype="<f4", count=n_verts * 3, offset=v_off).reshape(-1, 3)

    e_off, e_len = lump(12)
    edges = np.frombuffer(data, dtype="<u2", count=(e_len // 4) * 2, offset=e_off).reshape(-1, 2)

    se_off, se_len = lump(13)
    surfedges = np.frombuffer(data, dtype="<i4", count=se_len // 4, offset=se_off)

    f_off, f_len = lump(7)
    FACE_SIZE = 56
    n_faces = f_len // FACE_SIZE

    ti_off, ti_len = lump(6)
    td_off, td_len = lump(2)
    tst_off, tst_len = lump(44)
    tsd_off, tsd_len = lump(43)
    n_texinfo = ti_len // 72
    n_texdata = td_len // 32
    tex_offsets = np.frombuffer(data, dtype="<i4", count=tst_len // 4, offset=tst_off)
    tex_strings = data[tsd_off: tsd_off + tsd_len]
    td_nameids = [struct.unpack_from("<i", data, td_off + i * 32 + 12)[0] for i in range(n_texdata)]

    def _tex_name(texinfo_idx):
        if texinfo_idx < 0 or texinfo_idx >= n_texinfo:
            return ""
        td_idx = struct.unpack_from("<i", data, ti_off + texinfo_idx * 72 + 68)[0]
        if td_idx < 0 or td_idx >= n_texdata:
            return ""
        nid = td_nameids[td_idx]
        if nid < 0 or nid >= len(tex_offsets):
            return ""
        off = int(tex_offsets[nid])
        end = tex_strings.index(b"\x00", off)
        return tex_strings[off:end].decode("ascii", errors="replace").lower()

    all_v, all_f = [], []
    vert_offset = 0
    for fi in range(n_faces):
        fb = f_off + fi * FACE_SIZE
        side = data[fb + 2]
        firstedge = struct.unpack_from("<i", data, fb + 4)[0]
        numedges = struct.unpack_from("<h", data, fb + 8)[0]
        dispinfo = struct.unpack_from("<h", data, fb + 12)[0]
        texinfo = struct.unpack_from("<h", data, fb + 10)[0]

        if dispinfo != -1 or numedges < 3:
            continue
        if _tex_name(texinfo) in NON_SOLID_BSP:
            continue

        face_vidx = []
        for j in range(numedges):
            se = int(surfedges[firstedge + j])
            face_vidx.append(int(edges[se][0]) if se >= 0 else int(edges[-se][1]))
        if side == 0:
            face_vidx = face_vidx[::-1]

        local_v = verts[face_vidx]
        local_f = np.array([[0, j + 1, j + 2] for j in range(numedges - 2)], dtype=np.int32)
        all_v.append(local_v)
        all_f.append(local_f + vert_offset)
        vert_offset += len(local_v)

    if not all_v:
        raise ValueError(f"no solid geometry found in BSP: {path}")
    return np.concatenate(all_v), np.concatenate(all_f)


def _build_geometry_from_nav(path: str) -> tuple:
    with open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack_from("<I", data, 0)[0]
    assert magic == 0xFEEDFACE, f"Not a NAV file: {path}"
    places = struct.unpack_from("<H", data, 12)[0]
    n_areas = struct.unpack_from("<I", data, 14)[0]

    p = 18
    for _ in range(places):
        slen = struct.unpack_from("<H", data, p)[0]
        p += 2 + slen

    all_v, all_f = [], []
    vert_offset = 0
    for _ in range(n_areas):
        p += 6
        lx, ly, lz = struct.unpack_from("<fff", data, p)
        p += 12
        hx, hy, hz = struct.unpack_from("<fff", data, p)
        p += 12
        ne_z, sw_z = struct.unpack_from("<ff", data, p)
        p += 8
        mid_z = (ne_z + sw_z) / 2.0
        quad = np.array([
            [lx, ly, sw_z], [hx, ly, mid_z],
            [hx, hy, ne_z], [lx, hy, mid_z],
        ], dtype=np.float32)
        all_v.append(quad)
        all_f.append(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32) + vert_offset)
        vert_offset += 4

        for _ in range(4):
            cnt = struct.unpack_from("<I", data, p)[0]
            p += 4 + cnt * 4
        hs = data[p]
        p += 1 + hs * 17
        aa = data[p]
        p += 1 + aa * 14
        ep = struct.unpack_from("<I", data, p)[0]
        p += 4
        for _ in range(ep):
            p += 10
            sc = data[p]
            p += 1 + sc * 5
        p += 18

    if not all_v:
        raise ValueError(f"no walkable areas found in NAV: {path}")
    return np.concatenate(all_v), np.concatenate(all_f)


def _build_geometry(path: str) -> tuple:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".glb":
        return _build_geometry_from_glb(path)
    if ext == ".bsp":
        return _build_geometry_from_bsp(path)
    if ext == ".nav":
        return _build_geometry_from_nav(path)
    raise ValueError(f"unrecognised map format: {path}")


def _disk_cache_path(cache_dir: str, map_name: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in map_name)
    return Path(cache_dir) / f"{safe_name}.geometry.npz"


def _atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


def build_map_geometry(map_paths, *, disk_cache_dir: str | None, map_name: str) -> tuple:
    """Return (vertices, faces) for `map_name`, combining every path in `map_paths`.

    Checks the on-disk npz cache first (fast path: skips lump-table parsing and
    texture-based face filtering entirely). On a miss, parses the raw files via the
    ported format-specific readers and populates the cache for next time.
    """
    if disk_cache_dir is not None:
        cache_path = _disk_cache_path(disk_cache_dir, map_name)
        if cache_path.exists():
            data = np.load(cache_path)
            return data["vertices"], data["faces"]

    all_v, all_f = [], []
    vert_offset = 0
    for path in map_paths:
        v, f = _build_geometry(path)
        all_v.append(v)
        all_f.append(f + vert_offset)
        vert_offset += len(v)
    vertices = np.concatenate(all_v)
    faces = np.concatenate(all_f)

    if disk_cache_dir is not None:
        _atomic_write_npz(_disk_cache_path(disk_cache_dir, map_name),
                          vertices=vertices, faces=faces)
    return vertices, faces


def load_mesh_cached(map_name: str, map_paths, *, disk_cache_dir: str | None = ".kuration_cs2_collision_cache"):
    """Load (from the on-disk cache, or by parsing raw map files) a warmed trimesh.Trimesh
    for `map_name`. Callers should additionally hold an in-memory `{map_name: mesh}` dict
    (see `CollisionMechanics.__init__`) so this only runs once per map per process."""
    trimesh = _trimesh()
    vertices, faces = build_map_geometry(map_paths, disk_cache_dir=disk_cache_dir, map_name=map_name)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Warm the BVH once at load time rather than on the first ray cast.
    mesh.ray.intersects_location(ray_origins=[[0, 0, 0]], ray_directions=[[0, 0, 1]],
                                 multiple_hits=False)
    return mesh


# ── Proximity raycast, ported verbatim from collision_classifier.py ─────────────────

EYE_HEIGHT = 64.0
RAY_BACKOFF = 16.0
RAY_ANGLES_RAD = np.linspace(-math.pi / 2, math.pi / 2, 9)
_RAY_COS_OFF = np.cos(RAY_ANGLES_RAD)
_RAY_SIN_OFF = np.sin(RAY_ANGLES_RAD)
_RAY_ZEROS = np.zeros(len(RAY_ANGLES_RAD))


def cast_proximity_rays(mesh, origin: np.ndarray, yaw_deg: float, proximity: float):
    """Horizontal fan of 9 rays from eye position. Returns (hit, nearest_dist, wall_normal_xy)."""
    yaw_rad = math.radians(yaw_deg)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

    eye = origin.copy()
    eye[2] += EYE_HEIGHT

    dx = cy * _RAY_COS_OFF - sy * _RAY_SIN_OFF
    dy = sy * _RAY_COS_OFF + cy * _RAY_SIN_OFF
    dirs = np.column_stack((dx, dy, _RAY_ZEROS))

    origins = np.tile(eye, (len(dirs), 1))
    origins[:, 0] -= RAY_BACKOFF * dx
    origins[:, 1] -= RAY_BACKOFF * dy

    locs, idx_ray, idx_tri = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False)

    if len(locs) == 0:
        return False, math.inf, None

    dists = np.linalg.norm(locs - origins[idx_ray], axis=1)
    nearest_i = int(np.argmin(dists))
    raw_d = float(dists[nearest_i])
    nearest_d = max(0.0, raw_d - RAY_BACKOFF)

    if nearest_d > proximity:
        return False, nearest_d, None

    raw_n = mesh.face_normals[idx_tri[nearest_i]].copy()
    normal_xy = np.array([float(raw_n[0]), float(raw_n[1]), 0.0])
    n_len = float(np.linalg.norm(normal_xy))
    normal_xy = normal_xy / n_len if n_len > 1e-6 else None
    return True, nearest_d, normal_xy


def wall_stopped(vel_prev: np.ndarray, vel_curr: np.ndarray, wall_normal_xy,
                 min_approach: float, max_residual: float) -> tuple:
    """Returns (flagged, vn_prev_hus, vn_curr_hus). Ported verbatim from collision_classifier.py."""
    vp2 = vel_prev[:2]
    vc2 = vel_curr[:2]

    if wall_normal_xy is not None:
        n = wall_normal_xy[:2].copy()
        if float(np.dot(vp2, n)) > 0:
            n = -n
        vn_prev = max(0.0, float(-np.dot(vp2, n))) * 64
        vn_curr = abs(float(np.dot(vc2, n))) * 64
    else:
        vn_prev = float(np.linalg.norm(vp2)) * 64
        vn_curr = float(np.linalg.norm(vc2)) * 64

    flagged = vn_prev >= min_approach and vn_curr <= max_residual
    return flagged, vn_prev, vn_curr
