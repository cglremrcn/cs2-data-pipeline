"""
CS2 Map Geometry Extractor
Extracts nav mesh areas and computes wall boundaries from CS2 map VPK files.

Output format:
- nav_areas: All walkable area polygons with coordinates
- boundary_edges: Edges that are NOT shared between areas (= walls/map boundaries)
- area_connections: Which areas connect (shared edges)

Coordinate system: CS2 game coordinates (same as demo file player positions)
"""

import vpk
import json
import os
import sys
import struct
import math
from collections import defaultdict
from awpy import Nav


def extract_nav_from_vpk(vpk_path, map_name):
    """Extract .nav file from a map VPK."""
    pak = vpk.open(vpk_path)
    nav_path = f"maps/{map_name}.nav"
    if nav_path not in pak:
        raise FileNotFoundError(f"{nav_path} not found in {vpk_path}")
    return pak[nav_path].read()


def parse_nav_v36(nav_bytes):
    """Parse nav mesh v36 binary format (unsupported by awpy).

    v36 layout:
    - Header: magic(4) + version(4) + sub_version(4) + unknown(4) = 16 bytes
    - KV3 blob: 129 bytes
    - Corner count (4) + corners (12 * count)
    - Polygon count (4) + polygons: u8(vc) + u32[vc](indices) + u32(0xFFFFFFFF)
    """
    offset = 0
    magic, version, sub_version, unk = struct.unpack_from('<IIII', nav_bytes, offset)
    offset = 16

    if version < 36:
        raise ValueError(f"Expected v36+, got v{version}")

    # Skip KV3 blob (129 bytes)
    offset += 129

    # Read corners
    corner_count = struct.unpack_from('<I', nav_bytes, offset)[0]
    offset += 4
    corners = []
    for _ in range(corner_count):
        x, y, z = struct.unpack_from('<fff', nav_bytes, offset)
        corners.append((x, y, z))
        offset += 12

    # Read polygons
    poly_count = struct.unpack_from('<I', nav_bytes, offset)[0]
    offset += 4
    polygons = []
    for _ in range(poly_count):
        vc = nav_bytes[offset]
        offset += 1
        indices = []
        for _ in range(vc):
            idx = struct.unpack_from('<I', nav_bytes, offset)[0]
            indices.append(idx)
            offset += 4
        term = struct.unpack_from('<I', nav_bytes, offset)[0]
        offset += 4
        if term != 0xFFFFFFFF:
            raise ValueError(f"Bad polygon terminator: 0x{term:08X}")
        polygons.append(indices)

    # Build areas dict (same format as awpy output)
    areas = {}
    for i, poly_indices in enumerate(polygons):
        area_corners = [
            {"x": corners[idx][0], "y": corners[idx][1], "z": corners[idx][2]}
            for idx in poly_indices
        ]
        areas[i] = {
            "area_id": i,
            "hull_index": 0,
            "dynamic_attribute_flags": 0,
            "corners": area_corners,
            "connections": [],
            "ladders_above": [],
            "ladders_below": [],
        }

    return version, areas


def get_nav_version(nav_bytes):
    """Read just the version from nav binary data."""
    return struct.unpack_from('<I', nav_bytes, 4)[0]


def edge_key(p1, p2):
    """Create a canonical key for an edge (order-independent)."""
    a = (round(p1[0], 1), round(p1[1], 1), round(p1[2], 1))
    b = (round(p2[0], 1), round(p2[1], 1), round(p2[2], 1))
    return tuple(sorted([a, b]))


def compute_boundary_edges_from_areas(areas_dict):
    """
    Find edges that appear in only one area (boundary/wall edges).
    Works with both awpy Nav objects and our v36 parsed dict.
    """
    edge_count = defaultdict(list)

    for area_id, area in areas_dict.items():
        corners = area["corners"]
        n = len(corners)
        for i in range(n):
            c1, c2 = corners[i], corners[(i + 1) % n]
            p1 = (c1["x"], c1["y"], c1["z"])
            p2 = (c2["x"], c2["y"], c2["z"])
            ek = edge_key(p1, p2)
            edge_count[ek].append(area_id)

    boundary_edges = []
    for ek, area_ids in edge_count.items():
        if len(area_ids) == 1:
            p1, p2 = ek
            length = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            boundary_edges.append({
                "start": {"x": p1[0], "y": p1[1], "z": p1[2]},
                "end": {"x": p2[0], "y": p2[1], "z": p2[2]},
                "length": round(length, 2),
                "area_id": area_ids[0],
            })

    return boundary_edges


def process_map(vpk_path, map_name, output_dir):
    """Process a single map: extract nav mesh, compute geometry, save JSON."""
    print(f"\nProcessing {map_name}...")

    nav_data = extract_nav_from_vpk(vpk_path, map_name)
    version = get_nav_version(nav_data)

    if version >= 36:
        # Use our custom v36 parser
        nav_version, areas_dict = parse_nav_v36(nav_data)
        print(f"  Nav version: {nav_version} (custom parser), Areas: {len(areas_dict)}")
    else:
        # Use awpy for v35 and below
        temp_nav = os.path.join(output_dir, f"{map_name}.nav")
        with open(temp_nav, "wb") as f:
            f.write(nav_data)
        nav = Nav.from_path(temp_nav)
        nav_version = nav.version
        print(f"  Nav version: {nav_version}, Areas: {len(nav.areas)}")
        nav_dict = nav.to_dict()
        areas_dict = nav_dict["areas"]
        os.remove(temp_nav)

    # Compute boundary edges (walls)
    boundary_edges = compute_boundary_edges_from_areas(areas_dict)
    print(f"  Boundary edges (walls): {len(boundary_edges)}")

    areas_list = list(areas_dict.values())

    # Compute map bounds
    all_x, all_y, all_z = [], [], []
    for area in areas_list:
        for corner in area["corners"]:
            all_x.append(corner["x"])
            all_y.append(corner["y"])
            all_z.append(corner["z"])

    map_bounds = {
        "min": {"x": min(all_x), "y": min(all_y), "z": min(all_z)},
        "max": {"x": max(all_x), "y": max(all_y), "z": max(all_z)},
    }
    print(f"  Map bounds: X[{map_bounds['min']['x']:.0f}, {map_bounds['max']['x']:.0f}] "
          f"Y[{map_bounds['min']['y']:.0f}, {map_bounds['max']['y']:.0f}] "
          f"Z[{map_bounds['min']['z']:.0f}, {map_bounds['max']['z']:.0f}]")

    output = {
        "map_name": map_name,
        "coordinate_system": "CS2 game coordinates (same as demo file positions)",
        "nav_version": nav_version,
        "total_areas": len(areas_dict),
        "total_boundary_edges": len(boundary_edges),
        "map_bounds": map_bounds,
        "nav_areas": areas_list,
        "boundary_edges": boundary_edges,
    }

    out_path = os.path.join(output_dir, f"{map_name}_geometry.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")

    return output


def main():
    vpk_dir = "C:/Users/Emc/Desktop/cs2-data-pipeline/map_vpks"
    output_dir = "C:/Users/Emc/Desktop/cs2-data-pipeline/map_geometry"
    os.makedirs(output_dir, exist_ok=True)

    # Map names to process
    target_maps = sys.argv[1:] if len(sys.argv) > 1 else ["de_dust2", "de_mirage"]

    for map_name in target_maps:
        vpk_path = os.path.join(vpk_dir, f"{map_name}.vpk")
        if not os.path.exists(vpk_path):
            print(f"Skipping {map_name}: VPK not found at {vpk_path}")
            continue
        try:
            process_map(vpk_path, map_name, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
