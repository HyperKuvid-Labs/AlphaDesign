#!/usr/bin/env python3
"""Showcase render of the (fixed) ultra-realistic F1 front wing.

By default, generates a wing with IDEAL_F1_PARAMETERS (mirror-symmetric
about Y=0 after the half-span generation fix) and renders it headless with
pyrender/EGL. Pass --stl to render an existing STL instead (e.g. a specific
optimizer-found design).

Usage:
    PYOPENGL_PLATFORM=egl python3 src/experiments/render_wing_showcase.py [accent|carbon]
    PYOPENGL_PLATFORM=egl python3 src/experiments/render_wing_showcase.py \\
        --stl path/to/design.stl --out paper/figures/my_render.png [accent|carbon]

Hero renders of the GA best design (paper/figures/best_design_render_*.png)
use --filter-small-components, which renders only the 4 wing elements and
2 endplates and omits the ~320 tiny unstitched strake/VG/fence/pylon
patches that z-fight into visible clutter at the footplate:

    PYOPENGL_PLATFORM=egl python3 src/experiments/render_wing_showcase.py carbon \
        --stl f1_wing_output/best_design_phase_two_ga_only.stl \
        --filter-small-components --width 2560 --height 1440 \
        --out paper/figures/best_design_render_carbon.png

Output:
    paper/figures/wing_showcase.png (default) or --out path
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alphadesign.wing_generator import (  # noqa: E402
    UltraRealisticF1FrontWingGenerator,
    IDEAL_F1_PARAMETERS,
)

import trimesh  # noqa: E402
import pyrender  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STL_CACHE = os.path.join(REPO_ROOT, "f1_wing_output", "showcase_ideal.stl")
OUT_PATH = os.path.join(REPO_ROOT, "paper", "figures", "wing_showcase.png")

BACKGROUND = (0xF8 / 255, 0xF5 / 255, 0xED / 255, 1.0)  # #f8f5ed


def generate_or_load_mesh(stl_path=None):
    """Load an existing STL, or generate the IDEAL preset (cached), in meters."""
    if stl_path is None:
        stl_path = STL_CACHE
        if not os.path.exists(stl_path):
            gen = UltraRealisticF1FrontWingGenerator(**IDEAL_F1_PARAMETERS)
            wing = gen.generate_complete_wing(os.path.basename(stl_path))
            if wing is None:
                raise RuntimeError("wing generation failed")
    tm = trimesh.load(stl_path)
    tm.apply_scale(0.001)  # mm -> m
    return tm


def filter_small_components(tm, min_extent_m):
    """Keep only connected components whose largest bounding-box extent is
    >= ``min_extent_m``. Used for hero renders only: the generator emits the
    ~20 ancillary features (strakes, vortex generators, fences, pylons,
    cascades) as hundreds of unstitched thin patches that z-fight at most
    camera angles. The 4 wing elements and the 2 endplates are by far the
    largest components, so an extent threshold isolates them cleanly."""
    # Two criteria: largest extent >= min_extent_m AND second-largest extent
    # >= min_extent_m / 3. The second test rejects ribbon-like parts (the
    # 640 x 74 x 12 mm serrated footplate strips are *longer* than the
    # 411 x 252 mm endplates, so a single length threshold cannot separate
    # them, but their width cannot pass a plate-like test).
    comps = tm.split(only_watertight=False)
    keep = []
    for c in comps:
        ext = np.sort(c.extents)
        if ext[-1] >= min_extent_m and ext[-2] >= min_extent_m / 3.0:
            keep.append(c)
    print(
        f"component filter: kept {len(keep)}/{len(comps)} components "
        f"(max extent >= {min_extent_m * 1000:.0f} mm, 2nd extent >= {min_extent_m * 1000 / 3:.0f} mm); "
        f"kept extents (mm): {sorted(tuple(np.round(np.sort(c.extents)[::-1] * 1000).astype(int)) for c in keep)}"
    )
    return trimesh.util.concatenate(keep)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera pose matrix looking from eye to target."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up2
    pose[:3, 2] = -fwd
    pose[:3, 3] = eye
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", choices=["carbon", "accent"], default="carbon")
    parser.add_argument("--stl", default=None, help="render an existing STL instead of the IDEAL preset")
    parser.add_argument("--out", default=None, help="output PNG path")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--filter-small-components", nargs="?", const=300.0, default=None, type=float,
        metavar="MIN_EXTENT_MM",
        help="render only connected components whose largest bbox extent is >= this "
             "(mm; default 300 when given without a value). Drops the tiny unstitched "
             "strake/VG/fence/pylon patches that z-fight; keeps elements + endplates.",
    )
    parser.add_argument(
        "--view", type=float, nargs=3, default=(-1.0, 0.72, 0.52), metavar=("X", "Y", "Z"),
        help="camera direction from the bbox center (unnormalised); default front-quarter",
    )
    args = parser.parse_args()

    tm = generate_or_load_mesh(args.stl)
    if args.filter_small_components is not None:
        tm = filter_small_components(tm, args.filter_small_components / 1000.0)
    bounds = tm.bounds  # meters
    center = tm.centroid
    span_y = bounds[1][1] - bounds[0][1]  # dominant dimension (~1.9-2.5 m)
    print(f"mesh: {len(tm.vertices)} verts, {len(tm.faces)} faces")
    print(f"bounds (m): {bounds}, span_y={span_y:.3f}")

    # --- material ---
    if args.variant == "accent":
        # flat orange accent matching the repo's figure palette (#d94210)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0xD9 / 255, 0x42 / 255, 0x10 / 255, 1.0),
            metallicFactor=0.05,
            roughnessFactor=0.55,
            doubleSided=True,
        )
        out_path = args.out or OUT_PATH.replace(".png", "_accent.png")
    else:
        # dark carbon-fiber-ish (these wings are carbon fiber)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.09, 0.09, 0.105, 1.0),
            metallicFactor=0.60,
            roughnessFactor=0.30,
            doubleSided=True,
        )
        out_path = args.out or OUT_PATH
    mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=BACKGROUND, ambient_light=(0.18, 0.18, 0.19))
    scene.add(mesh)

    # --- camera: framed off the spanwise (Y) extent, not a bounding sphere ---
    fov_y = np.radians(33.0)
    aspect = 16 / 9
    # horizontal half-angle (span is along the image's horizontal axis)
    half_w = span_y / 2 * 1.06
    dist_w = half_w / (np.tan(fov_y / 2) * aspect)
    half_h = max(bounds[1][2] - bounds[0][2], bounds[1][0] - bounds[0][0]) / 2 * 1.15
    dist_h = half_h / np.tan(fov_y / 2)
    dist = max(dist_w, dist_h) + 0.25

    view_dir = np.array(args.view, dtype=float)  # default: front-quarter view from above
    view_dir /= np.linalg.norm(view_dir)
    target = (bounds[0] + bounds[1]) / 2  # frame on bbox center, not centroid
    eye = target + view_dir * dist
    cam = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=aspect)
    scene.add(cam, pose=look_at(eye, target))

    # --- lighting: warm key, cool fill, rim ---
    key = pyrender.DirectionalLight(color=(1.0, 0.95, 0.88), intensity=3.2)
    scene.add(key, pose=look_at(center + np.array([2.5, -2.0, 3.0]), center))

    fill = pyrender.DirectionalLight(color=(0.72, 0.82, 1.0), intensity=1.1)
    scene.add(fill, pose=look_at(center + np.array([-2.0, 2.5, 1.2]), center))

    rim = pyrender.DirectionalLight(color=(1.0, 0.98, 0.95), intensity=1.8)
    scene.add(rim, pose=look_at(center + np.array([-2.5, 0.5, 2.5]), center))

    r = pyrender.OffscreenRenderer(args.width, args.height)
    color, _ = r.render(scene)
    r.delete()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import matplotlib.image as mpimg
    mpimg.imsave(out_path, color)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
