"""
Biological Mesh Generators
===========================

Pure numpy functions that generate high-fidelity triangle mesh data for
biological entities in the diapedesis scene. Each function returns a tuple:

    (vertices, face_vertex_counts, face_vertex_indices, normals)

- vertices: np.ndarray (N, 3) float32 — mesh vertex positions
- face_vertex_counts: np.ndarray (F,) int32 — vertices per face (always 3)
- face_vertex_indices: np.ndarray (F*3,) int32 — triangle indices
- normals: np.ndarray (N, 3) float32 — per-vertex normals

All meshes are unit-scale (diameter ≈ 1.0) and centered at origin.
Scale to biological size at reference time in the scene builder.

No USD imports — this module is pure geometry math (numpy only).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# Type alias for mesh data tuple
MeshData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# ── Utility Functions ──────────────────────────────────────────────────


def _compute_vertex_normals(
    vertices: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Compute smooth per-vertex normals from face normals."""
    normals = np.zeros_like(vertices)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    # Accumulate face normals to vertices
    for k in range(3):
        np.add.at(normals, triangles[:, k], face_normals)
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return (normals / norms).astype(np.float32)


def _make_mesh_data(
    vertices: np.ndarray, triangles: np.ndarray
) -> MeshData:
    """Convert vertices + triangle indices to USD-compatible mesh data."""
    vertices = vertices.astype(np.float32)
    normals = _compute_vertex_normals(vertices, triangles)
    face_counts = np.full(len(triangles), 3, dtype=np.int32)
    face_indices = triangles.flatten().astype(np.int32)
    return vertices, face_counts, face_indices, normals


def _uv_sphere(
    n_lat: int = 16, n_lon: int = 32, radius: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate UV sphere vertices and triangle indices."""
    verts = []
    # Top pole
    verts.append([0, radius, 0])
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        y = radius * math.cos(theta)
        r = radius * math.sin(theta)
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            verts.append([r * math.cos(phi), y, r * math.sin(phi)])
    # Bottom pole
    verts.append([0, -radius, 0])
    verts = np.array(verts, dtype=np.float32)

    tris = []
    # Top cap
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        tris.append([0, 1 + j, 1 + j_next])
    # Middle strips
    for i in range(n_lat - 2):
        row = 1 + i * n_lon
        next_row = row + n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            a, b = row + j, row + j_next
            c, d = next_row + j, next_row + j_next
            tris.append([a, c, b])
            tris.append([b, c, d])
    # Bottom cap
    bottom = len(verts) - 1
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        tris.append([last_row + j, bottom, last_row + j_next])

    return verts, np.array(tris, dtype=np.int32)


def _cylinder(
    n_segments: int = 16, radius: float = 0.5, height: float = 1.0,
    caps: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate cylinder vertices and triangles along Y axis."""
    verts = []
    tris = []
    half_h = height / 2

    # Side vertices: two rings
    for ring in range(2):
        y = -half_h if ring == 0 else half_h
        for j in range(n_segments):
            phi = 2 * math.pi * j / n_segments
            verts.append([radius * math.cos(phi), y, radius * math.sin(phi)])

    # Side faces
    for j in range(n_segments):
        j_next = (j + 1) % n_segments
        a, b = j, j_next
        c, d = n_segments + j, n_segments + j_next
        tris.append([a, c, b])
        tris.append([b, c, d])

    if caps:
        # Bottom cap
        center_bot = len(verts)
        verts.append([0, -half_h, 0])
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            tris.append([center_bot, j_next, j])
        # Top cap
        center_top = len(verts)
        verts.append([0, half_h, 0])
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            tris.append([center_top, n_segments + j, n_segments + j_next])

    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


def _capsule(
    n_segments: int = 16, n_cap_rings: int = 8,
    radius: float = 0.5, height: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate capsule (cylinder + hemisphere caps) along Y axis."""
    half_h = height / 2
    verts = []
    tris = []

    # Top pole
    verts.append([0, half_h + radius, 0])

    # Top hemisphere rings
    for i in range(1, n_cap_rings + 1):
        theta = (math.pi / 2) * i / n_cap_rings
        y = half_h + radius * math.cos(theta)
        r = radius * math.sin(theta)
        for j in range(n_segments):
            phi = 2 * math.pi * j / n_segments
            verts.append([r * math.cos(phi), y, r * math.sin(phi)])

    # Cylinder bottom ring
    for j in range(n_segments):
        phi = 2 * math.pi * j / n_segments
        verts.append([radius * math.cos(phi), -half_h, radius * math.sin(phi)])

    # Bottom hemisphere rings
    for i in range(1, n_cap_rings):
        theta = (math.pi / 2) + (math.pi / 2) * i / n_cap_rings
        y = -half_h + radius * math.cos(theta)
        r = radius * math.sin(theta)
        for j in range(n_segments):
            phi = 2 * math.pi * j / n_segments
            verts.append([r * math.cos(phi), y, r * math.sin(phi)])

    # Bottom pole
    verts.append([0, -half_h - radius, 0])
    verts = np.array(verts, dtype=np.float32)

    # Triangulate: top cap
    for j in range(n_segments):
        j_next = (j + 1) % n_segments
        tris.append([0, 1 + j, 1 + j_next])

    # All middle ring strips
    total_rings = n_cap_rings + 1 + (n_cap_rings - 1)  # top hemi + cyl bottom + bot hemi
    for ring_idx in range(total_rings - 1):
        row = 1 + ring_idx * n_segments
        next_row = row + n_segments
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            a, b = row + j, row + j_next
            c, d = next_row + j, next_row + j_next
            tris.append([a, c, b])
            tris.append([b, c, d])

    # Bottom cap
    bottom = len(verts) - 1
    last_row = 1 + (total_rings - 1) * n_segments
    for j in range(n_segments):
        j_next = (j + 1) % n_segments
        tris.append([last_row + j, bottom, last_row + j_next])

    return verts, np.array(tris, dtype=np.int32)


def _ellipsoid(
    n_lat: int = 12, n_lon: int = 16,
    rx: float = 0.5, ry: float = 0.5, rz: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ellipsoid vertices and triangles."""
    verts, tris = _uv_sphere(n_lat, n_lon, 1.0)
    verts[:, 0] *= rx
    verts[:, 1] *= ry
    verts[:, 2] *= rz
    return verts, tris


def _merge_meshes(
    mesh_list: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge multiple (verts, tris) into one mesh."""
    all_verts = []
    all_tris = []
    offset = 0
    for verts, tris in mesh_list:
        all_verts.append(verts)
        all_tris.append(tris + offset)
        offset += len(verts)
    return (
        np.concatenate(all_verts, axis=0),
        np.concatenate(all_tris, axis=0),
    )


def _translate_mesh(
    verts: np.ndarray, dx: float, dy: float, dz: float
) -> np.ndarray:
    """Return translated copy of vertices."""
    out = verts.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    out[:, 2] += dz
    return out


def _scale_mesh(
    verts: np.ndarray, sx: float, sy: float, sz: float
) -> np.ndarray:
    """Return scaled copy of vertices."""
    out = verts.copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    out[:, 2] *= sz
    return out


def _simple_noise(x: float, y: float, z: float, seed: int = 0) -> float:
    """Simple deterministic pseudo-noise (hash-based, -1..1)."""
    h = seed
    h = (h * 1103515245 + int(x * 73856093) + 12345) & 0x7FFFFFFF
    h = (h * 1103515245 + int(y * 19349663) + 12345) & 0x7FFFFFFF
    h = (h * 1103515245 + int(z * 83492791) + 12345) & 0x7FFFFFFF
    return (h % 10000) / 5000.0 - 1.0


# ── Mesh Generators ───────────────────────────────────────────────────


def generate_rbc_mesh(lod: int = 0) -> MeshData:
    """Evans-Fung (1972) biconcave disc.

    Unit diameter (D=1.0). The parametric profile:
        z(r) = (D/2)*sqrt(1 - 4r²/D²) * (0.0518 + 2.0026*(2r/D)² - 4.491*(2r/D)⁴)

    Real RBC: D=7.5μm, rim=2.2μm, center=1.0μm. Scale at reference time.
    """
    n_lon = 32 if lod == 0 else 16
    n_rad = 16 if lod == 0 else 8
    D = 1.0
    half_D = D / 2

    verts = []
    # Center point (dimple)
    verts.append([0.0, 0.0, 0.0])

    # Radial rings from center outward
    for i in range(1, n_rad + 1):
        r_frac = i / n_rad  # 0..1
        r = r_frac * half_D
        rn = 2 * r / D  # normalized radius
        rn2 = rn * rn
        rn4 = rn2 * rn2
        sq = 1.0 - 4 * r * r / (D * D)
        if sq <= 0:
            z = 0.0
        else:
            z = half_D * math.sqrt(sq) * (0.0518 + 2.0026 * rn2 - 4.491 * rn4)

        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            # Top surface
            verts.append([x, z, y])

    # Mirror for bottom surface (skip center — shared)
    for i in range(1, n_rad + 1):
        r_frac = i / n_rad
        r = r_frac * half_D
        rn = 2 * r / D
        rn2 = rn * rn
        rn4 = rn2 * rn2
        sq = 1.0 - 4 * r * r / (D * D)
        if sq <= 0:
            z = 0.0
        else:
            z = -half_D * math.sqrt(sq) * (0.0518 + 2.0026 * rn2 - 4.491 * rn4)

        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            verts.append([x, z, y])

    verts = np.array(verts, dtype=np.float32)

    tris = []

    # Top surface: center fan
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        tris.append([0, 1 + j, 1 + j_next])

    # Top surface: ring strips
    for i in range(n_rad - 1):
        row = 1 + i * n_lon
        next_row = row + n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            a, b = row + j, row + j_next
            c, d = next_row + j, next_row + j_next
            tris.append([a, c, b])
            tris.append([b, c, d])

    # Bottom surface: center fan
    bot_start = 1 + n_rad * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        tris.append([0, bot_start + j_next, bot_start + j])

    # Bottom surface: ring strips
    for i in range(n_rad - 1):
        row = bot_start + i * n_lon
        next_row = row + n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            a, b = row + j, row + j_next
            c, d = next_row + j, next_row + j_next
            tris.append([a, d, b])
            tris.append([a, c, d])

    # Stitch outer rim (top outer ring to bottom outer ring)
    top_outer = 1 + (n_rad - 1) * n_lon
    bot_outer = bot_start + (n_rad - 1) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        a = top_outer + j
        b = top_outer + j_next
        c = bot_outer + j
        d = bot_outer + j_next
        tris.append([a, b, d])
        tris.append([a, d, c])

    tris = np.array(tris, dtype=np.int32)
    return _make_mesh_data(verts, tris)


def generate_neutrophil_body_mesh(lod: int = 0) -> MeshData:
    """Neutrophil body: sphere with organic noise + microvilli protrusions.

    Unit diameter. LOD0: ~4000 tris with microvilli. LOD1: smooth sphere.
    """
    n_lat = 24 if lod == 0 else 12
    n_lon = 32 if lod == 0 else 16
    verts, tris = _uv_sphere(n_lat, n_lon, 0.5)

    if lod == 0:
        # Apply organic noise displacement to surface
        for i in range(len(verts)):
            x, y, z = verts[i]
            r = math.sqrt(x * x + y * y + z * z)
            if r < 1e-6:
                continue
            nx, ny, nz = x / r, y / r, z / r
            # Low-frequency noise for organic irregularity
            noise = _simple_noise(x * 5, y * 5, z * 5, seed=42) * 0.02
            # Higher frequency for membrane texture
            noise += _simple_noise(x * 15, y * 15, z * 15, seed=99) * 0.008
            new_r = r + noise
            verts[i] = [nx * new_r, ny * new_r, nz * new_r]

        # Add microvilli: small conical bumps on surface
        # Use Poisson-disc-like distribution (golden angle spiral)
        n_microvilli = 60
        golden_angle = math.pi * (3 - math.sqrt(5))
        mv_meshes = []
        for k in range(n_microvilli):
            # Fibonacci sphere sampling
            y_frac = 1 - (k / (n_microvilli - 1)) * 2  # -1..1
            r_ring = math.sqrt(1 - y_frac * y_frac)
            theta = golden_angle * k
            nx = r_ring * math.cos(theta)
            ny = y_frac
            nz = r_ring * math.sin(theta)

            # Position on surface
            surface_r = 0.5
            px, py, pz = nx * surface_r, ny * surface_r, nz * surface_r

            # Create tiny cone (3 tris) protruding outward
            tip = np.array([[
                px + nx * 0.06,
                py + ny * 0.06,
                pz + nz * 0.06
            ]], dtype=np.float32)

            # Tangent vectors for cone base
            if abs(ny) < 0.9:
                up = np.array([0, 1, 0])
            else:
                up = np.array([1, 0, 0])
            n_vec = np.array([nx, ny, nz])
            t1 = np.cross(n_vec, up)
            t1 = t1 / (np.linalg.norm(t1) + 1e-10)
            t2 = np.cross(n_vec, t1)

            base_r = 0.012
            n_sides = 4
            base_verts = []
            for s in range(n_sides):
                angle = 2 * math.pi * s / n_sides
                bp = (np.array([px, py, pz])
                      + t1 * base_r * math.cos(angle)
                      + t2 * base_r * math.sin(angle))
                base_verts.append(bp)
            base_verts = np.array(base_verts, dtype=np.float32)

            mv_verts = np.concatenate([tip, base_verts], axis=0)
            mv_tris = []
            for s in range(n_sides):
                s_next = (s + 1) % n_sides
                mv_tris.append([0, 1 + s, 1 + s_next])
            mv_tris = np.array(mv_tris, dtype=np.int32)
            mv_meshes.append((mv_verts, mv_tris))

        # Merge body + microvilli
        all_meshes = [(verts, tris)] + mv_meshes
        verts, tris = _merge_meshes(all_meshes)

    return _make_mesh_data(verts, tris)


def generate_neutrophil_nucleus_mesh(lod: int = 0) -> MeshData:
    """Multi-lobed neutrophil nucleus (3-4 connected lobes).

    Uses overlapping ellipsoids merged into a single mesh.
    Lobe positions match diapedesis_scene.py line 603-606.
    Unit scale — scale at reference time.
    """
    # Lobe positions (relative, unit scale)
    lobe_positions = [
        (0.15, 0.1, 0.0),
        (-0.15, -0.05, 0.1),
        (0.0, -0.1, -0.15),
        (0.1, 0.15, -0.05),
    ]

    n_lat = 10 if lod == 0 else 6
    n_lon = 14 if lod == 0 else 8

    meshes = []
    for lp in lobe_positions:
        v, t = _ellipsoid(n_lat, n_lon, 0.22, 0.18, 0.16)
        v = _translate_mesh(v, lp[0], lp[1], lp[2])
        meshes.append((v, t))

    # Add connecting bridges between adjacent lobes
    if lod == 0:
        for i in range(len(lobe_positions) - 1):
            p1 = lobe_positions[i]
            p2 = lobe_positions[i + 1]
            mid = [(p1[k] + p2[k]) / 2 for k in range(3)]
            v, t = _ellipsoid(6, 8, 0.08, 0.08, 0.08)
            v = _translate_mesh(v, mid[0], mid[1], mid[2])
            meshes.append((v, t))

    verts, tris = _merge_meshes(meshes)
    return _make_mesh_data(verts, tris)


def generate_selectin_mesh(lod: int = 0) -> Dict[str, MeshData]:
    """Selectin lollipop: SCR stalk beads + EGF + lectin head.

    Returns dict with "stalk" and "lectin_head" sub-meshes for
    separate material binding.

    Unit height (~1.0). Scale at reference time.
    """
    if lod == 1:
        # Simple capsule
        v, t = _capsule(8, 4, 0.08, 0.6)
        stalk = _make_mesh_data(v, t)
        v_h, t_h = _ellipsoid(6, 8, 0.15, 0.10, 0.15)
        v_h = _translate_mesh(v_h, 0, 0.5, 0)
        head = _make_mesh_data(v_h, t_h)
        return {"stalk": stalk, "lectin_head": head}

    # LOD0: detailed beaded stalk
    stalk_meshes = []

    # 5 SCR domain beads
    for k in range(5):
        v, t = _uv_sphere(8, 10, 0.04)
        v = _translate_mesh(v, 0, k * 0.12 + 0.05, 0)
        stalk_meshes.append((v, t))

    # Thin connecting rods between beads
    for k in range(4):
        v, t = _cylinder(6, 0.015, 0.08, caps=False)
        v = _translate_mesh(v, 0, k * 0.12 + 0.09, 0)
        stalk_meshes.append((v, t))

    # EGF domain (slightly larger bead)
    v, t = _uv_sphere(8, 10, 0.05)
    v = _translate_mesh(v, 0, 0.65, 0)
    stalk_meshes.append((v, t))

    stalk_v, stalk_t = _merge_meshes(stalk_meshes)
    stalk_data = _make_mesh_data(stalk_v, stalk_t)

    # Lectin head (oblate spheroid)
    v_h, t_h = _ellipsoid(10, 14, 0.12, 0.07, 0.12)
    v_h = _translate_mesh(v_h, 0, 0.78, 0)
    head_data = _make_mesh_data(v_h, t_h)

    return {"stalk": stalk_data, "lectin_head": head_data}


def generate_icam1_mesh(lod: int = 0) -> Dict[str, MeshData]:
    """ICAM-1: 5 Ig-domain bead-rod with kink at domain 3.

    Returns dict with "body" and "tip" sub-meshes.
    Kink angle=0.4 rad matches diapedesis_scene.py line 495.
    """
    kink_angle = 0.4

    if lod == 1:
        # Simple kinked cylinder
        v, t = _capsule(8, 4, 0.06, 0.4)
        return {"body": _make_mesh_data(v, t), "tip": _make_mesh_data(v, t)}

    body_meshes = []
    tip_mesh = None

    for k in range(5):
        is_tip = (k == 4)
        y = k * 0.13
        x = 0.0
        if k >= 3:
            x = (k - 2) * 0.13 * math.sin(kink_angle)
            y = 0.26 + (k - 2) * 0.13 * math.cos(kink_angle)

        v, t = _ellipsoid(8, 10, 0.035, 0.05, 0.035)
        v = _translate_mesh(v, x, y, 0)

        if is_tip:
            tip_mesh = (v, t)
        else:
            body_meshes.append((v, t))

    # Connecting rods
    positions = []
    for k in range(5):
        y = k * 0.13
        x = 0.0
        if k >= 3:
            x = (k - 2) * 0.13 * math.sin(kink_angle)
            y = 0.26 + (k - 2) * 0.13 * math.cos(kink_angle)
        positions.append((x, y))

    for k in range(4):
        p1 = positions[k]
        p2 = positions[k + 1]
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dx, dy)
        v, t = _cylinder(6, 0.012, length, caps=False)
        # Rotate to align with segment
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i in range(len(v)):
            ox, oy = v[i, 0], v[i, 1]
            v[i, 0] = ox * cos_a + oy * sin_a
            v[i, 1] = -ox * sin_a + oy * cos_a
        v = _translate_mesh(v, mx, my, 0)
        body_meshes.append((v, t))

    body_v, body_t = _merge_meshes(body_meshes)
    tip_v, tip_t = tip_mesh

    return {
        "body": _make_mesh_data(body_v, body_t),
        "tip": _make_mesh_data(tip_v, tip_t),
    }


def generate_pecam1_mesh(lod: int = 0) -> MeshData:
    """PECAM-1 dimer pair: two 6-bead antiparallel rods.

    Tips interdigitate in the middle. Single mesh.
    """
    n_beads = 6 if lod == 0 else 3
    n_lat = 8 if lod == 0 else 5
    n_lon = 10 if lod == 0 else 6
    bead_r = 0.03
    spacing = 0.10

    meshes = []
    # Rod 1: beads along +Y
    for k in range(n_beads):
        v, t = _ellipsoid(n_lat, n_lon, bead_r, bead_r * 1.3, bead_r)
        v = _translate_mesh(v, 0, (0.12 + k * spacing), 0)
        meshes.append((v, t))
    # Rod 2: beads along -Y (antiparallel)
    for k in range(n_beads):
        v, t = _ellipsoid(n_lat, n_lon, bead_r, bead_r * 1.3, bead_r)
        v = _translate_mesh(v, 0, -(0.12 + k * spacing), 0)
        meshes.append((v, t))

    verts, tris = _merge_meshes(meshes)
    return _make_mesh_data(verts, tris)


def generate_ecoli_mesh(lod: int = 0) -> Dict[str, MeshData]:
    """E. coli bacterium: capsule body + flagella + C3b complement coating.

    Returns dict with "body", "flagella", "complement" sub-meshes.
    """
    # Body: capsule
    n_seg = 16 if lod == 0 else 8
    n_cap = 8 if lod == 0 else 4
    body_v, body_t = _capsule(n_seg, n_cap, 0.15, 0.5)
    body_data = _make_mesh_data(body_v, body_t)

    if lod == 1:
        return {"body": body_data, "flagella": body_data, "complement": body_data}

    # Flagella: 4 helical tubes
    flagella_meshes = []
    rng = np.random.RandomState(42)
    for f in range(4):
        # Helix path along -Y from one end
        path_points = []
        helix_r = 0.06
        helix_pitch = 0.15
        n_points = 30
        base_angle = rng.uniform(0, 2 * math.pi)
        for p in range(n_points):
            t = p / n_points
            y = -0.35 - t * 0.5
            angle = base_angle + t * 6 * math.pi
            x = helix_r * math.cos(angle)
            z = helix_r * math.sin(angle)
            path_points.append([x, y, z])

        # Sweep a small circle along path
        tube_r = 0.008
        n_tube_seg = 4
        tube_verts = []
        tube_tris = []
        for p_idx in range(len(path_points)):
            pt = path_points[p_idx]
            # Approximate tangent
            if p_idx < len(path_points) - 1:
                nxt = path_points[p_idx + 1]
                tangent = np.array([nxt[k] - pt[k] for k in range(3)])
            else:
                prv = path_points[p_idx - 1]
                tangent = np.array([pt[k] - prv[k] for k in range(3)])
            tangent = tangent / (np.linalg.norm(tangent) + 1e-10)

            # Build normal/binormal
            if abs(tangent[1]) < 0.9:
                up = np.array([0, 1, 0])
            else:
                up = np.array([1, 0, 0])
            n1 = np.cross(tangent, up)
            n1 = n1 / (np.linalg.norm(n1) + 1e-10)
            n2 = np.cross(tangent, n1)

            for s in range(n_tube_seg):
                angle = 2 * math.pi * s / n_tube_seg
                offset = n1 * tube_r * math.cos(angle) + n2 * tube_r * math.sin(angle)
                tube_verts.append([pt[0] + offset[0], pt[1] + offset[1], pt[2] + offset[2]])

        tube_verts = np.array(tube_verts, dtype=np.float32)
        for p_idx in range(len(path_points) - 1):
            row = p_idx * n_tube_seg
            next_row = row + n_tube_seg
            for s in range(n_tube_seg):
                s_next = (s + 1) % n_tube_seg
                a, b = row + s, row + s_next
                c, d = next_row + s, next_row + s_next
                tube_tris.append([a, c, b])
                tube_tris.append([b, c, d])
        tube_tris = np.array(tube_tris, dtype=np.int32)
        flagella_meshes.append((tube_verts, tube_tris))

    flag_v, flag_t = _merge_meshes(flagella_meshes)
    flagella_data = _make_mesh_data(flag_v, flag_t)

    # Complement C3b coating: 20 small bumps on body surface
    c3b_meshes = []
    for k in range(20):
        theta = rng.uniform(0, 2 * math.pi)
        y = rng.uniform(-0.2, 0.2)
        r_body = 0.15
        x = r_body * math.cos(theta) * 1.05
        z = r_body * math.sin(theta) * 1.05
        v, t = _uv_sphere(5, 6, 0.025)
        v = _translate_mesh(v, x, y, z)
        c3b_meshes.append((v, t))

    c3b_v, c3b_t = _merge_meshes(c3b_meshes)
    complement_data = _make_mesh_data(c3b_v, c3b_t)

    return {"body": body_data, "flagella": flagella_data, "complement": complement_data}


def generate_macrophage_mesh(lod: int = 0) -> MeshData:
    """Macrophage: amoeboid sphere with 3-5 pseudopod extensions.

    Unit diameter. Irregular organic surface.
    """
    n_lat = 20 if lod == 0 else 10
    n_lon = 28 if lod == 0 else 14
    verts, tris = _uv_sphere(n_lat, n_lon, 0.5)

    if lod == 0:
        # Pseudopod directions (3-5 extensions)
        rng = np.random.RandomState(77)
        n_pods = 4
        pod_dirs = []
        for _ in range(n_pods):
            theta = rng.uniform(0, 2 * math.pi)
            phi = rng.uniform(0.3, math.pi - 0.3)
            dx = math.sin(phi) * math.cos(theta)
            dy = math.cos(phi)
            dz = math.sin(phi) * math.sin(theta)
            pod_dirs.append(np.array([dx, dy, dz]))

        # Displace vertices toward pseudopod directions
        for i in range(len(verts)):
            x, y, z = verts[i]
            r = math.sqrt(x * x + y * y + z * z)
            if r < 1e-6:
                continue
            n_vec = np.array([x / r, y / r, z / r])

            # Check proximity to pseudopod directions
            max_influence = 0.0
            for pd in pod_dirs:
                dot = float(np.dot(n_vec, pd))
                if dot > 0.6:
                    influence = (dot - 0.6) / 0.4
                    max_influence = max(max_influence, influence)

            # Extend pseudopod vertices outward
            extension = max_influence * 0.2
            # Also add organic noise
            noise = _simple_noise(x * 8, y * 8, z * 8, seed=55) * 0.015
            new_r = r + extension + noise
            verts[i] = [n_vec[0] * new_r, n_vec[1] * new_r, n_vec[2] * new_r]

    return _make_mesh_data(verts, tris)


def generate_endothelial_cell_mesh(lod: int = 0) -> MeshData:
    """Flat hexagonal tile with slight dome for endothelial cell.

    Tessellates vessel inner wall. Raised edges at junctions.
    """
    n_sides = 6
    outer_r = 0.5
    dome_height = 0.03
    edge_height = 0.02 if lod == 0 else 0

    verts = [[0, dome_height, 0]]  # Center (domed up)
    # Outer ring
    for j in range(n_sides):
        angle = 2 * math.pi * j / n_sides
        x = outer_r * math.cos(angle)
        z = outer_r * math.sin(angle)
        verts.append([x, edge_height, z])

    if lod == 0:
        # Add mid-ring for smoother dome
        mid_r = outer_r * 0.55
        for j in range(n_sides):
            angle = 2 * math.pi * (j + 0.5) / n_sides
            x = mid_r * math.cos(angle)
            z = mid_r * math.sin(angle)
            verts.append([x, dome_height * 0.8, z])

    verts = np.array(verts, dtype=np.float32)
    tris = []

    if lod == 0:
        # Center to mid-ring
        mid_start = 1 + n_sides
        for j in range(n_sides):
            tris.append([0, mid_start + j, mid_start + (j + 1) % n_sides])
        # Mid-ring to outer ring
        for j in range(n_sides):
            j_next = (j + 1) % n_sides
            m = mid_start + j
            m_next = mid_start + j_next
            o = 1 + j
            o_next = 1 + j_next
            tris.append([m, o, m_next])
            tris.append([m_next, o, o_next])
    else:
        # Simple fan
        for j in range(n_sides):
            j_next = (j + 1) % n_sides
            tris.append([0, 1 + j, 1 + j_next])

    # Bottom face (mirror)
    n_existing = len(verts)
    bottom_verts = verts.copy()
    bottom_verts[:, 1] *= -1
    verts = np.concatenate([verts, bottom_verts], axis=0)

    bottom_tris = []
    for tri in tris:
        bottom_tris.append([tri[0] + n_existing, tri[2] + n_existing, tri[1] + n_existing])
    tris = tris + bottom_tris

    # Side faces connecting top and bottom outer rings
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        top_a = 1 + j
        top_b = 1 + j_next
        bot_a = n_existing + 1 + j
        bot_b = n_existing + 1 + j_next
        tris.append([top_a, top_b, bot_b])
        tris.append([top_a, bot_b, bot_a])

    tris = np.array(tris, dtype=np.int32)
    return _make_mesh_data(verts, tris)


def generate_fibrin_fiber_mesh(lod: int = 0) -> MeshData:
    """Fibrin fiber: thin cylinder with random waviness.

    Unit length along Y. Scale length and radius at reference time.
    """
    n_seg = 12 if lod == 0 else 6
    n_points = 20 if lod == 0 else 8

    rng = np.random.RandomState(42)
    radius = 0.02
    half_h = 0.5

    # Path with waviness
    path = []
    for i in range(n_points):
        t = i / (n_points - 1)
        y = -half_h + t * 2 * half_h
        wx = rng.normal(0, 0.01) if lod == 0 else 0
        wz = rng.normal(0, 0.01) if lod == 0 else 0
        path.append([wx, y, wz])

    # Sweep circle along path
    verts = []
    for pt in path:
        for j in range(n_seg):
            angle = 2 * math.pi * j / n_seg
            x = pt[0] + radius * math.cos(angle)
            z = pt[2] + radius * math.sin(angle)
            verts.append([x, pt[1], z])

    verts = np.array(verts, dtype=np.float32)

    tris = []
    for i in range(n_points - 1):
        row = i * n_seg
        next_row = row + n_seg
        for j in range(n_seg):
            j_next = (j + 1) % n_seg
            a, b = row + j, row + j_next
            c, d = next_row + j, next_row + j_next
            tris.append([a, c, b])
            tris.append([b, c, d])

    # Caps
    center_bot = len(verts)
    verts = list(verts)
    verts.append(path[0])
    for j in range(n_seg):
        j_next = (j + 1) % n_seg
        tris.append([center_bot, j_next, j])

    center_top = len(verts)
    verts.append(path[-1])
    last_row = (n_points - 1) * n_seg
    for j in range(n_seg):
        j_next = (j + 1) % n_seg
        tris.append([center_top, last_row + j, last_row + j_next])

    verts = np.array(verts, dtype=np.float32)
    tris = np.array(tris, dtype=np.int32)
    return _make_mesh_data(verts, tris)


def generate_collagen_helix_mesh(lod: int = 0) -> MeshData:
    """Collagen triple-helix: three intertwined helical tubes.

    Unit length along Y. Classic collagen rope structure.
    """
    n_tube_seg = 5 if lod == 0 else 3
    n_points = 30 if lod == 0 else 10
    tube_r = 0.015 if lod == 0 else 0.025
    helix_r = 0.04
    n_strands = 3
    half_h = 0.5

    strand_meshes = []
    for strand in range(n_strands):
        phase = strand * 2 * math.pi / n_strands
        path_points = []
        for p in range(n_points):
            t = p / (n_points - 1)
            y = -half_h + t * 2 * half_h
            angle = phase + t * 4 * math.pi
            x = helix_r * math.cos(angle)
            z = helix_r * math.sin(angle)
            path_points.append([x, y, z])

        # Sweep tube along path
        tube_verts = []
        for p_idx in range(len(path_points)):
            pt = path_points[p_idx]
            if p_idx < len(path_points) - 1:
                nxt = path_points[p_idx + 1]
                tangent = np.array([nxt[k] - pt[k] for k in range(3)])
            else:
                prv = path_points[p_idx - 1]
                tangent = np.array([pt[k] - prv[k] for k in range(3)])
            tangent = tangent / (np.linalg.norm(tangent) + 1e-10)

            if abs(tangent[1]) < 0.9:
                up = np.array([0, 1, 0])
            else:
                up = np.array([1, 0, 0])
            n1 = np.cross(tangent, up)
            n1 = n1 / (np.linalg.norm(n1) + 1e-10)
            n2 = np.cross(tangent, n1)

            for s in range(n_tube_seg):
                angle = 2 * math.pi * s / n_tube_seg
                offset = n1 * tube_r * math.cos(angle) + n2 * tube_r * math.sin(angle)
                tube_verts.append([
                    pt[0] + offset[0], pt[1] + offset[1], pt[2] + offset[2]
                ])

        tube_verts = np.array(tube_verts, dtype=np.float32)
        tube_tris = []
        for p_idx in range(len(path_points) - 1):
            row = p_idx * n_tube_seg
            next_row = row + n_tube_seg
            for s in range(n_tube_seg):
                s_next = (s + 1) % n_tube_seg
                a, b = row + s, row + s_next
                c, d = next_row + s, next_row + s_next
                tube_tris.append([a, c, b])
                tube_tris.append([b, c, d])
        tube_tris = np.array(tube_tris, dtype=np.int32)
        strand_meshes.append((tube_verts, tube_tris))

    verts, tris = _merge_meshes(strand_meshes)
    return _make_mesh_data(verts, tris)
