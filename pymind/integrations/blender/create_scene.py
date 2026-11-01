"""Run inside Blender to create a JSON-described scene and export it as GLB.

Example:
blender --background --python pymind/integrations/blender/create_scene.py -- \
  --spec examples/blender_scene.json --output output/scene.glb
"""

import argparse
import json
from pathlib import Path

from pymind.integrations.blender.scene_spec import validate_scene_spec


def create_scene(spec, output_path):
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "This command must run with Blender's bundled Python: "
            "blender --background --python ..."
        ) from exc

    objects = validate_scene_spec(spec)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for item in objects:
        if item.kind == "cube":
            bpy.ops.mesh.primitive_cube_add(size=1, location=item.location)
        elif item.kind == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16,
                                                 radius=0.5, location=item.location)
        else:
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.5,
                                                depth=1, location=item.location)

        obj = bpy.context.object
        obj.name = item.name
        obj.scale = item.scale
        material = bpy.data.materials.new(name=f"{item.name}_Material")
        material.diffuse_color = item.color
        obj.data.materials.append(material)

    # Add a camera and light so the exported scene is immediately inspectable.
    bpy.ops.object.camera_add(location=(7, -9, 6))
    camera = bpy.context.object
    camera.name = "PyMind_Camera"
    camera.rotation_euler = (1.05, 0, 0.67)
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(4, -4, 7))
    light = bpy.context.object
    light.name = "PyMind_Key_Light"
    light.data.energy = 1200
    light.data.shape = "DISK"
    light.data.size = 5

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_apply=True,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to a JSON scene specification")
    parser.add_argument("--output", default="output/pymind_scene.glb")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    output = create_scene(spec, args.output)
    print(f"Exported GLB: {output}")


if __name__ == "__main__":
    main()
