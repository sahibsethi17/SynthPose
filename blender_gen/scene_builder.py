"""Scene construction and domain randomisation.

Notes on Blender 5.2, which differs from the 4.x API most tutorials assume:

* the EEVEE engine id is ``BLENDER_EEVEE`` again (4.2-4.5 used ``BLENDER_EEVEE_NEXT``);
* ``Scene.node_tree`` is gone -- the compositor is now an explicitly assigned
  node group on ``Scene.compositing_node_group``;
* the File Output node uses ``directory``/``file_name``/``file_output_items``
  instead of ``base_path``/``file_slots``, and its format is locked to
  multilayer OpenEXR, which is why depth lands in an ``.exr`` with a ``depth.V``
  channel rather than a plain single-channel image.
"""

from __future__ import annotations

import colorsys

import bpy

#: Written into every depth EXR wherever no geometry was hit.
DEPTH_BACKGROUND = 1e10


def _rand_rgb(rng, sat=(0.0, 0.7), val=(0.15, 0.95)):
    h, s, v = rng.random(), rng.uniform(*sat), rng.uniform(*val)
    return (*colorsys.hsv_to_rgb(h, s, v), 1.0)


def reset_scene():
    """Delete all objects and orphaned datablocks from the previous sample.

    Rebuilding from empty each iteration costs a little time but removes the
    whole class of bugs where a stale light or material leaks into later renders.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def configure_render(resolution=256, engine="BLENDER_EEVEE", samples=32):
    """Apply render settings and enable the Z pass."""
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    else:
        scene.eevee.taa_render_samples = samples

    scene.view_layers[0].use_pass_z = True
    return scene


def setup_depth_output(directory):
    """Wire Render Layers -> File Output for the Z pass; return the output node."""
    scene = bpy.context.scene
    for group in list(bpy.data.node_groups):
        if group.name == "DepthComp":
            bpy.data.node_groups.remove(group)

    tree = bpy.data.node_groups.new("DepthComp", "CompositorNodeTree")
    scene.compositing_node_group = tree

    layers = tree.nodes.new("CompositorNodeRLayers")
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.file_output_items.new("FLOAT", "depth")   # must exist before format is set
    out.directory = str(directory)
    out.format.color_depth = "32"
    tree.links.new(layers.outputs["Depth"], out.inputs["depth"])
    return out


def add_camera(rng, lens_range=(40.0, 80.0)):
    """Create the scene camera with a randomised focal length.

    Varying focal length (and therefore K) stops the model from memorising a
    single fixed projection, which is what makes the depth/scale ambiguity
    inherent to monocular pose visible rather than hidden by a constant.
    """
    data = bpy.data.cameras.new("Camera")
    data.lens = rng.uniform(*lens_range)
    data.sensor_fit = "AUTO"
    cam = bpy.data.objects.new("Camera", data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def apply_material(obj, rng):
    """Give ``obj`` a randomised Principled BSDF and return its parameters."""
    mat = bpy.data.materials.new("RandomMat")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")

    params = {
        "base_color": _rand_rgb(rng, sat=(0.25, 0.95), val=(0.12, 0.85)),
        "roughness": float(rng.uniform(0.15, 0.95)),
        "metallic": float(rng.random() < 0.25) * float(rng.uniform(0.6, 1.0)),
    }
    bsdf.inputs["Base Color"].default_value = params["base_color"]
    bsdf.inputs["Roughness"].default_value = params["roughness"]
    bsdf.inputs["Metallic"].default_value = params["metallic"]

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return params


def randomise_lighting(rng):
    """One sun plus 1-2 area lights at random positions, with a random world colour."""
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = _rand_rgb(rng, sat=(0.0, 0.35), val=(0.05, 0.6))
    bg.inputs["Strength"].default_value = float(rng.uniform(0.2, 1.0))

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = float(rng.uniform(2.5, 7.0))
    sun_data.color = _rand_rgb(rng, sat=(0.0, 0.25), val=(0.85, 1.0))[:3]
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (rng.uniform(0, 1.1), rng.uniform(-0.6, 0.6), rng.uniform(0, 6.28))
    bpy.context.scene.collection.objects.link(sun)

    for i in range(rng.integers(1, 3)):
        data = bpy.data.lights.new(f"Area{i}", type="AREA")
        data.energy = float(rng.uniform(120.0, 500.0))
        data.size = float(rng.uniform(1.0, 4.0))
        data.color = _rand_rgb(rng, sat=(0.0, 0.4), val=(0.7, 1.0))[:3]
        lamp = bpy.data.objects.new(f"Area{i}", data)
        lamp.location = tuple(rng.uniform(-7, 7, size=3) * [1, 1, 0.6] + [0, 0, 2.5])
        lamp.rotation_euler = tuple(rng.uniform(-1.2, 1.2, size=3))
        bpy.context.scene.collection.objects.link(lamp)


def place_camera(cam, cam_to_world, geo):
    """Move ``cam`` to a 4x4 camera-to-world matrix, and verify it took effect.

    Assigning ``cam.matrix_world`` from a nested Python list appears to succeed but
    is silently discarded -- ``matrix_world`` is derived, so the subsequent depsgraph
    update recomputes it from the (untouched) loc/rot/scale channels and the camera
    never moves. Nothing raises; the renders just come out from the wrong viewpoint.
    Setting the underlying channels explicitly is unambiguous, and the assertion
    makes any future regression loud rather than silent.
    """
    cam.location = tuple(float(v) for v in cam_to_world[:3, 3])
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = geo.quat_from_matrix(cam_to_world[:3, :3]).tolist()
    bpy.context.view_layer.update()

    import numpy as np
    actual = np.array(cam.matrix_world).reshape(4, 4)
    assert np.allclose(actual, cam_to_world, atol=1e-4), "camera pose did not take effect"
    return cam
