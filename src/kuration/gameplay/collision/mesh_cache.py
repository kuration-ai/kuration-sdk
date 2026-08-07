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

The on-disk cache is versioned and source-aware: every `.npz` records both the
`GEOMETRY_CACHE_VERSION` it was written under and a fingerprint of the raw geometry paths,
sizes, and modification times. A mismatched entry is rebuilt instead of being trusted.
The parser version protects code changes, while the source fingerprint prevents replacing
or editing a map under the same stable map name from silently serving stale geometry.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
from pathlib import Path

import numpy as np

# Bump whenever any geometry parser below changes its output for the same raw file --
# on-disk cache entries written under a different version are re-parsed, not trusted.
# Version 1 is the first versioned format; unversioned caches (the pre-SDK plugin layout)
# are treated as stale.
GEOMETRY_CACHE_VERSION = 2

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

# sizeof(ddispinfo_t) / sizeof(CDispVert), byte-offset layout verified against
# source-sdk-2013's public/bspfile.h and cross-checked against a real compiled BSP (a map's
# DISPINFO count * 176 landing exactly on its DISP_VERTS lump size for every displacement's
# (2**power+1)**2 vertex count).
DISPINFO_SIZE = 176
DISPVERT_SIZE = 20

STATIC_PROP_NAME_LENGTH = 128
# On-disk byte order of the GAMELUMP_STATIC_PROPS four-CC id. The C enum spells it 'sprp' as a
# multi-char int literal, which serializes byte-reversed on a little-endian disk -- confirmed by
# reading an actual BSP's game lump directory rather than assumed.
_STATIC_PROP_FOURCC = b"prps"

# Static-prop model-name keywords this collision mesh cares about, each mapped to a generic
# (dx, dy, dz) fallback box size in Hammer units. prop_static covers hundreds of purely
# decorative models too (windows, trims, rooftops); restricting to these keywords keeps the
# added proxy geometry to the categories that were actually reported missing (doors, crates,
# barrels, fence/gate/cover-style barriers) instead of guessing boxes for everything.
_PROP_KEYWORDS = {
    "door": (48.0, 8.0, 108.0),
    "crate": (48.0, 48.0, 48.0),
    "barrel": (24.0, 24.0, 40.0),
    "fence": (96.0, 8.0, 72.0),
    "gate": (96.0, 8.0, 72.0),
    "cover": (96.0, 8.0, 72.0),
    "rail": (96.0, 8.0, 42.0),
    "cart": (48.0, 72.0, 48.0),
}
# Many of this content set's filenames literally embed their dimensions, e.g.
# "dust_crate_style_01_37x37x74.mdl" or "dust_door_46x106_06.mdl" -- prefer that over the
# keyword's generic fallback whenever it's present, since we have no access to the actual .mdl
# geometry/bounding box to measure this precisely.
_PROP_DIM_RE = re.compile(r"(\d{2,3})x(\d{2,3})(?:x(\d{2,3}))?")


def _trimesh():
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            'the collision operator needs trimesh (and rtree for its ray BVH): '
            'pip install "kuration-sdk[collision]"'
        ) from e
    return trimesh


def _read_glb(glb_path: str):
    with open(glb_path, "rb") as f:
        magic, version, _ = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"not a GLB 2.0 file: {glb_path}")
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


def _bsp_lump(data: bytes, idx: int) -> tuple:
    off_hdr = 8 + idx * 16
    ofs = struct.unpack_from("<i", data, off_hdr)[0]
    length = struct.unpack_from("<i", data, off_hdr + 4)[0]
    return ofs, length


def _read_dispinfos(data: bytes) -> list:
    """DISPINFO-lump-order list of (startPosition, DispVertStart, power) -- a face's own
    `dispinfo` field (dface_t, offset 12) is a direct index into this list, so callers don't
    need `ddispinfo_t.m_iMapFace` (the reverse pointer) at all."""
    ofs, length = _bsp_lump(data, 26)
    n = length // DISPINFO_SIZE
    out = []
    for i in range(n):
        base = ofs + i * DISPINFO_SIZE
        sx, sy, sz, vert_start, _tri_start, power = struct.unpack_from("<fffiii", data, base)
        out.append((np.array([sx, sy, sz], dtype=np.float64), vert_start, power))
    return out


def _read_disp_verts(data: bytes) -> np.ndarray:
    """(N, 4) array of (vx, vy, vz, dist) per CDispVert -- the trailing alpha field is
    rendering-only and dropped."""
    ofs, length = _bsp_lump(data, 33)
    n = length // DISPVERT_SIZE
    raw = np.frombuffer(data, dtype="<f4", count=n * 5, offset=ofs).reshape(-1, 5)
    return raw[:, :4]


def _displacement_triangles(corners: np.ndarray, start_position: np.ndarray, vert_start: int,
                             power: int, disp_verts: np.ndarray) -> tuple:
    """Build the (2**power + 1)**2 vertex grid for one displacement surface and triangulate it.

    Ported from CCoreDispInfo::GenerateDispSurf (source-sdk-2013's public/builddisp.cpp): rotate
    the base face's 4 corners so index 0 is whichever corner sits closest to `start_position`
    (CCoreDispSurface::FindSurfPointStartIndex), bilinear-interpolate the flat grid across that
    quad, then displace each grid vertex along its own recorded (direction, distance) pair.
    Neighbor-based crack stitching between adjacent displacements is not reproduced -- harmless
    here since this mesh only ever gets used for proximity raycasting, never rendering.
    """
    start_idx = int(np.argmin(np.linalg.norm(corners - start_position, axis=1)))
    p = np.roll(corners, -start_idx, axis=0)  # p[0] is now the corner closest to start_position

    n = (1 << power) + 1
    oo = 1.0 / (n - 1)
    edge0 = (p[1] - p[0]) * oo
    edge1 = (p[2] - p[3]) * oo

    verts = np.empty((n * n, 3), dtype=np.float64)
    for i in range(n):
        end0 = p[0] + edge0 * i
        end1 = p[3] + edge1 * i
        seg = (end1 - end0) * oo
        for j in range(n):
            idx = i * n + j
            dvert = disp_verts[vert_start + idx]
            verts[idx] = end0 + seg * j + dvert[:3] * dvert[3]

    tri = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            tri.append((a, b, c))
            tri.append((b, d, c))
    return verts, np.array(tri, dtype=np.int32)


def _prop_box_size(model_name: str):
    """(dx, dy, dz) box size for a static-prop model, or None if it isn't one of the
    door/crate/barrel/barrier-ish keyword categories this collision mesh cares about."""
    full = model_name.lower()
    fallback = None
    for kw, size in _PROP_KEYWORDS.items():
        if kw in full:
            fallback = size
            break
    if fallback is None:
        return None
    fname = full.rsplit("/", 1)[-1]
    m = _PROP_DIM_RE.search(fname)
    if m:
        a, b, c = m.groups()
        if c is not None:
            return (float(a), float(b), float(c))
        # Two-number names are door/window-style WxH panels -- borrow the keyword's thickness.
        return (float(a), fallback[1], float(b))
    return fallback


def _box_geometry(origin: np.ndarray, yaw_deg: float, size_xyz: tuple) -> tuple:
    """8-vertex/12-triangle solid box for a static prop, footprint rotated by yaw only (pitch/
    roll ignored -- placed clutter props are overwhelmingly yaw-only). `origin` is treated as
    the box's base-center: the standard placement convention for floor-standing prop content."""
    dx, dy, dz = size_xyz
    hx, hy = dx / 2.0, dy / 2.0
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    local = np.array([
        [-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0],
        [-hx, -hy, dz], [hx, -hy, dz], [hx, hy, dz], [-hx, hy, dz],
    ])
    rot = np.array([[cy, -sy], [sy, cy]])
    verts = local.copy()
    verts[:, :2] = local[:, :2] @ rot.T
    verts += np.asarray(origin, dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 5, 1], [0, 4, 5],  # front (-y)
        [1, 6, 2], [1, 5, 6],  # right (+x)
        [2, 7, 3], [2, 6, 7],  # back (+y)
        [3, 4, 0], [3, 7, 4],  # left (-x)
    ], dtype=np.int32)
    return verts, faces


def _read_static_props(data: bytes) -> list:
    """Parse the GAMELUMP_STATIC_PROPS ('sprp', on-disk b"prps") sub-lump: prop_static
    placements live in the binary game lump, not the plain-text entity lump (that only carries
    logic entities). The per-entry struct (StaticPropLumpV4_t..V6_t..current, see
    source-sdk-2013's public/gamebspfile.h) has grown fields across versions, but every version
    shares the same first 31 bytes (origin, angles, prop type, leaf range, solid byte). Reading
    just that stable prefix and deriving the per-entry stride from the sub-lump's own recorded
    byte length -- rather than hardcoding a version->size table -- works across whichever
    version a given BSP happens to use."""
    gl_off, _ = _bsp_lump(data, 35)
    lump_count = struct.unpack_from("<i", data, gl_off)[0]
    sprp = None
    p = gl_off + 4
    for _ in range(lump_count):
        gid, _flags, _version, fileofs, filelen = struct.unpack_from("<4sHHii", data, p)
        if gid == _STATIC_PROP_FOURCC:
            sprp = (fileofs, filelen)
        p += 16
    if sprp is None:
        return []

    ofs, filelen = sprp
    end = ofs + filelen
    p = ofs
    dict_count = struct.unpack_from("<i", data, p)[0]
    p += 4
    names = []
    for _ in range(dict_count):
        raw = data[p: p + STATIC_PROP_NAME_LENGTH]
        names.append(raw.split(b"\x00", 1)[0].decode("ascii", errors="replace"))
        p += STATIC_PROP_NAME_LENGTH

    leaf_count = struct.unpack_from("<i", data, p)[0]
    p += 4 + leaf_count * 2

    prop_count = struct.unpack_from("<i", data, p)[0]
    p += 4
    if prop_count == 0:
        return []
    stride = (end - p) / prop_count
    if stride != int(stride):
        # Sub-lump byte accounting doesn't divide evenly -- one of the lump/dict/leaf offsets
        # above is wrong. Bail out rather than read a garbage stride and silently misparse.
        return []
    stride = int(stride)

    props = []
    for i in range(prop_count):
        base = p + i * stride
        ox, oy, oz, _ap, ay, _ar, ptype, _first_leaf, _leaf_count, solid = struct.unpack_from(
            "<ffffffHHHB", data, base)
        props.append({
            "origin": np.array([ox, oy, oz], dtype=np.float64),
            "yaw": ay,
            "model": names[ptype] if 0 <= ptype < len(names) else "",
            "solid": solid,
        })
    return props


def _static_prop_geometry(data: bytes) -> tuple:
    all_v, all_f = [], []
    vert_offset = 0
    for prop in _read_static_props(data):
        if prop["solid"] == 0:
            continue
        size = _prop_box_size(prop["model"])
        if size is None:
            continue
        verts, faces = _box_geometry(prop["origin"], prop["yaw"], size)
        all_v.append(verts)
        all_f.append(faces + vert_offset)
        vert_offset += len(verts)
    if not all_v:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    return np.concatenate(all_v), np.concatenate(all_f)


def _build_geometry_from_bsp(path: str) -> tuple:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"VBSP":
        raise ValueError(f"not a VBSP file: {path}")

    def lump(idx):
        return _bsp_lump(data, idx)

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

    dispinfos = _read_dispinfos(data)
    disp_verts = _read_disp_verts(data) if dispinfos else None

    all_v, all_f = [], []
    vert_offset = 0
    for fi in range(n_faces):
        fb = f_off + fi * FACE_SIZE
        side = data[fb + 2]
        firstedge = struct.unpack_from("<i", data, fb + 4)[0]
        numedges = struct.unpack_from("<h", data, fb + 8)[0]
        dispinfo = struct.unpack_from("<h", data, fb + 12)[0]
        texinfo = struct.unpack_from("<h", data, fb + 10)[0]

        if numedges < 3:
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

        if dispinfo != -1:
            # Displacement base face: previously skipped outright, which is exactly why walls
            # built from displacements (common in this content set for rubble/sloped terrain-
            # like geometry) were invisible to collision raycasting. A displacement's base face
            # is always a quad (numedges == 4); anything else would mean our face/dispinfo
            # parsing has gone wrong somewhere, so skip defensively rather than guess.
            if numedges != 4:
                continue
            start_position, vert_start, power = dispinfos[dispinfo]
            local_v, local_f = _displacement_triangles(
                local_v.astype(np.float64), start_position, vert_start, power, disp_verts)
        else:
            local_f = np.array([[0, j + 1, j + 2] for j in range(numedges - 2)], dtype=np.int32)

        all_v.append(local_v)
        all_f.append(local_f + vert_offset)
        vert_offset += len(local_v)

    prop_v, prop_f = _static_prop_geometry(data)
    if len(prop_v):
        all_v.append(prop_v)
        all_f.append(prop_f + vert_offset)
        vert_offset += len(prop_v)

    if not all_v:
        raise ValueError(f"no solid geometry found in BSP: {path}")
    return np.concatenate(all_v), np.concatenate(all_f)


def _build_geometry_from_nav(path: str) -> tuple:
    with open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 0xFEEDFACE:
        raise ValueError(f"not a NAV file: {path}")
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


def _geometry_source_fingerprint(map_paths) -> str:
    """Stable identity for the exact raw geometry inputs used to build a cache entry."""
    sources = []
    for raw_path in map_paths:
        path = Path(raw_path).resolve()
        stat = path.stat()
        sources.append({
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return json.dumps(sources, sort_keys=True, separators=(",", ":"))


def _atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


def build_map_geometry(map_paths, *, disk_cache_dir: str | None, map_name: str) -> tuple:
    """Return (vertices, faces) for `map_name`, combining every path in `map_paths`.

    Checks the on-disk npz cache first (fast path: skips lump-table parsing and
    texture-based face filtering entirely). On a miss -- including a hit written under a
    different `GEOMETRY_CACHE_VERSION`, which means the parsers have changed since it was
    written -- parses the raw files via the ported format-specific readers and repopulates
    the cache for next time.
    """
    source_fingerprint = _geometry_source_fingerprint(map_paths)
    if disk_cache_dir is not None:
        cache_path = _disk_cache_path(disk_cache_dir, map_name)
        if cache_path.exists():
            with np.load(cache_path) as data:
                if ("cache_version" in data
                        and int(data["cache_version"]) == GEOMETRY_CACHE_VERSION
                        and "source_fingerprint" in data
                        and str(data["source_fingerprint"]) == source_fingerprint):
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
                          vertices=vertices, faces=faces,
                          cache_version=np.int64(GEOMETRY_CACHE_VERSION),
                          source_fingerprint=np.asarray(source_fingerprint))
    return vertices, faces


def load_mesh_cached(map_name: str, map_paths, *, disk_cache_dir: str | None = ".kuration_collision_cache"):
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


# ── Proximity raycast, ported from collision_classifier.py -- now cast at three heights ─────

EYE_HEIGHT = 64.0
# ~half of EYE_HEIGHT: catches barriers/props too short to reach a ray fan cast only at eye
# level. A fan cast only at EYE_HEIGHT sails clean over anything shorter than that -- this was
# the root cause of missed "waist-level barrier" collisions even where the barrier's geometry
# was otherwise present in the mesh.
WAIST_HEIGHT = 32.0
# Below even WAIST_HEIGHT: a real de_dust2 rubble/curb-style ledge was measured (by sweeping
# ray height at the actual player position/tick of a confirmed real collision) to be ~17-26 HU
# tall -- short enough that a 32 HU ray still sails clean over it. This third, low layer is
# sized to sit under that measured range.
LEDGE_HEIGHT = 16.0
RAY_HEIGHTS = (EYE_HEIGHT, WAIST_HEIGHT, LEDGE_HEIGHT)
RAY_BACKOFF = 16.0
RAY_ANGLES_RAD = np.linspace(-math.pi / 2, math.pi / 2, 9)
_RAY_COS_OFF = np.cos(RAY_ANGLES_RAD)
_RAY_SIN_OFF = np.sin(RAY_ANGLES_RAD)
_RAY_ZEROS = np.zeros(len(RAY_ANGLES_RAD))


def _cast_fan_at_height(mesh, origin: np.ndarray, cy: float, sy: float, height: float) -> tuple:
    """One horizontal 9-ray fan at a fixed height above `origin`. Returns (nearest_dist,
    wall_normal_xy) with no proximity clamping -- that happens once, across both height layers,
    in cast_proximity_rays."""
    eye = origin.copy()
    eye[2] += height

    dx = cy * _RAY_COS_OFF - sy * _RAY_SIN_OFF
    dy = sy * _RAY_COS_OFF + cy * _RAY_SIN_OFF
    dirs = np.column_stack((dx, dy, _RAY_ZEROS))

    origins = np.tile(eye, (len(dirs), 1))
    origins[:, 0] -= RAY_BACKOFF * dx
    origins[:, 1] -= RAY_BACKOFF * dy

    locs, idx_ray, idx_tri = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False)
    if len(locs) == 0:
        return math.inf, None

    dists = np.linalg.norm(locs - origins[idx_ray], axis=1)
    nearest_i = int(np.argmin(dists))
    nearest_d = max(0.0, float(dists[nearest_i]) - RAY_BACKOFF)

    raw_n = mesh.face_normals[idx_tri[nearest_i]].copy()
    normal_xy = np.array([float(raw_n[0]), float(raw_n[1]), 0.0])
    n_len = float(np.linalg.norm(normal_xy))
    normal_xy = normal_xy / n_len if n_len > 1e-6 else None
    return nearest_d, normal_xy


def cast_proximity_rays(mesh, origin: np.ndarray, yaw_deg: float, proximity: float):
    """Horizontal fan of 9 rays from eye position, repeated at a lower waist-level height and
    merged to the nearer hit. Returns (hit, nearest_dist, wall_normal_xy)."""
    yaw_rad = math.radians(yaw_deg)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

    best_d, best_n = math.inf, None
    for height in RAY_HEIGHTS:
        d, n = _cast_fan_at_height(mesh, origin, cy, sy, height)
        if d < best_d:
            best_d, best_n = d, n

    if best_d > proximity:
        return False, best_d, None
    return True, best_d, best_n


def wall_stopped(vel_prev: np.ndarray, vel_curr: np.ndarray, wall_normal_xy,
                 min_approach: float, max_residual: float) -> tuple:
    """Returns (flagged, vn_prev_hus, vn_curr_hus).

    `vel_prev`/`vel_curr` must already be true HU/s velocities -- this function does not
    guess a sample-rate conversion. The reference collision_classifier.py derived velocity
    from raw position deltas and multiplied by a hardcoded 64 to reach HU/s; that constant
    implicitly assumed a specific tick-spacing that does not hold for every dataset (it
    silently overstated velocity ~3-4x on real TeaPearce data, confirmed against that
    dataset's own recorded m_vecVelocity-derived vel_mag field). Callers now supply
    already-correctly-scaled velocity directly -- e.g. an adapter with a native recorded
    velocity field reports it as-is, while an adapter that only has positions derives
    velocity itself, at its own known true sample rate, before this function ever sees it.
    """
    vp2 = vel_prev[:2]
    vc2 = vel_curr[:2]

    if wall_normal_xy is not None:
        n = wall_normal_xy[:2].copy()
        if float(np.dot(vp2, n)) > 0:
            n = -n
        vn_prev = max(0.0, float(-np.dot(vp2, n)))
        vn_curr = abs(float(np.dot(vc2, n)))
    else:
        vn_prev = float(np.linalg.norm(vp2))
        vn_curr = float(np.linalg.norm(vc2))

    flagged = vn_prev >= min_approach and vn_curr <= max_residual
    return flagged, vn_prev, vn_curr
