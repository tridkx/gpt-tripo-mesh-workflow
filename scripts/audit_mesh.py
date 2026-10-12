"""Read-only mesh inventory for Blender 4.x/5.x, using only bundled bpy.

blender --background --python audit_mesh.py -- --input asset.blend --output audit.json

Inspect raw meshes, not evaluated modifiers. Areas are world-space triangle areas.
Component counts use shared vertices; UV bounds outside 0..1 are diagnostic only
(UDIM/tiling may be intended). Does NOT test UV overlap, distortion, equivalence,
material correctness, or pose. Does not save the input scene or change images.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import sys

import bpy


def components(mesh):
    parent = list(range(len(mesh.vertices)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in mesh.edges:
        a, b = map(find, e.vertices)
        parent[b] = a
    sizes = {}
    for i in range(len(parent)):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


def audit(obj, relative_area_epsilon):
    m = obj.data
    m.calc_loop_triangles()
    coords = [obj.matrix_world @ v.co for v in m.vertices]
    finite = [p for p in coords if all(math.isfinite(x) for x in p)]
    bounds = [[min(p[i] for p in finite) for i in range(3)],
              [max(p[i] for p in finite) for i in range(3)]] if finite else None
    diagonal2 = sum((bounds[1][i] - bounds[0][i]) ** 2 for i in range(3)) if bounds else 0
    threshold = diagonal2 * relative_area_epsilon
    area = 0.0
    degenerate = nonfinite_triangles = 0
    for t in m.loop_triangles:
        a, b, c = [coords[i] for i in t.vertices]
        value = (b - a).cross(c - a).length * 0.5
        if not math.isfinite(value):
            nonfinite_triangles += 1
        else:
            area += value
            degenerate += value <= threshold
    uv_layers = []
    for layer in m.uv_layers:
        values = [tuple(d.uv) for d in layer.data]
        valid = [p for p in values if all(math.isfinite(x) for x in p)]
        uv_bounds = [[min(p[i] for p in valid) for i in range(2)],
                     [max(p[i] for p in valid) for i in range(2)]] if valid else None
        collapsed = 0
        for t in m.loop_triangles:
            a, b, c = [values[j] for j in t.loops]
            uv_area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
            collapsed += math.isfinite(uv_area) and uv_area <= 1e-14
        uv_layers.append({"name": layer.name, "bounds": uv_bounds,
                          "nonfinite_corners": len(values) - len(valid),
                          "outside_unit_square_corners": sum(any(x < -1e-6 or x > 1 + 1e-6 for x in p) for p in valid),
                          "collapsed_triangles_at_1e-14": collapsed})
    materials = []
    for slot in obj.material_slots:
        mat = slot.material
        images = []
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type != 'TEX_IMAGE':
                    continue
                im = node.image
                images.append(None if im is None else {
                    "name": im.name, "size": list(im.size),
                    "has_data": bool(im.has_data), "packed": bool(im.packed_files),
                    "filepath": im.filepath, "colorspace": im.colorspace_settings.name})
        materials.append({"name": mat.name if mat else None, "direct_image_nodes": images})
    return {"name": obj.name, "vertices": len(m.vertices), "faces": len(m.polygons),
            "triangles": len(m.loop_triangles), "world_bounds": bounds,
            "world_area": area, "nonfinite_vertices": len(coords) - len(finite),
            "near_zero_triangles": degenerate, "world_area_threshold": threshold,
            "nonfinite_triangles": nonfinite_triangles,
            "connected_component_vertex_counts": components(m),
            "has_custom_normals": bool(m.has_custom_normals), "uv_layers": uv_layers,
            "vertex_groups": [g.name for g in obj.vertex_groups],
            "modifiers": [{"name": x.name, "type": x.type} for x in obj.modifiers],
            "materials": materials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--relative-area-epsilon', type=float, default=1e-14)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    source, target = Path(args.input).resolve(), Path(args.output).resolve()
    if source.suffix.lower() != '.blend' or not source.is_file():
        parser.error('--input must be an existing .blend file')
    if target == source or target.suffix.lower() != '.json':
        parser.error('--output must be a separate .json file')
    if not math.isfinite(args.relative_area_epsilon) or args.relative_area_epsilon < 0:
        parser.error('--relative-area-epsilon must be finite and nonnegative')
    bpy.ops.wm.open_mainfile(filepath=str(source))
    report = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "input": str(source), "blender": bpy.app.version_string,
              "scope": "raw scene meshes; direct image nodes only; no overlap/equivalence test",
              "armatures": [{"name": o.name, "bones": len(o.data.bones)} for o in bpy.context.scene.objects if o.type == 'ARMATURE'],
              "meshes": [audit(o, args.relative_area_epsilon) for o in bpy.context.scene.objects if o.type == 'MESH']}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('Audit written:', target, '; meshes:', len(report['meshes']))


if __name__ == '__main__':
    main()
