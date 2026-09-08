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
import math

import bpy
import numpy as np


def _cos(a):
    return math.cos(float(a))


def _sin(a):
    return math.sin(float(a))

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


#: Distractor primitives scattered around the target to create clutter.
_DISTRACTOR_OPS = ("primitive_cube_add", "primitive_uv_sphere_add", "primitive_cone_add",
                   "primitive_torus_add", "primitive_cylinder_add", "primitive_ico_sphere_add")


def randomise_world(rng):
    """Set the environment: a physical sky, procedural clutter, or a flat colour.

    Measuring the first dataset showed the model was nearly immune to what the generator
    randomised (lighting colour, +1.0 deg) and collapsed on the one thing it never varied
    (background, +83.9 deg). Real environments supply competing edges, texture and
    non-uniform illumination; a flat colour supplies none of it, so a model trained only
    on flat colour never learns to ignore any of it.

    A share of flat-colour worlds is kept deliberately, so the easy condition stays in the
    distribution and results remain comparable with the earlier runs.
    """
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    bg = tree.nodes["Background"]
    for node in list(tree.nodes):
        if node.type not in ("BACKGROUND", "OUTPUT_WORLD"):
            tree.nodes.remove(node)

    roll = rng.random()
    if roll < 0.45:                                   # physical sky
        sky = tree.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "MULTIPLE_SCATTERING" if rng.random() < 0.5 else "HOSEK_WILKIE"
        sky.sun_elevation = float(rng.uniform(0.05, 1.4))
        sky.sun_rotation = float(rng.uniform(0.0, 6.28))
        if hasattr(sky, "turbidity"):
            sky.turbidity = float(rng.uniform(1.5, 8.0))
        tree.links.new(sky.outputs[0], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = float(rng.uniform(0.25, 0.9))
        kind = "sky"
    elif roll < 0.80:                                 # procedural clutter
        noise = tree.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = float(rng.uniform(1.5, 14.0))
        if "Detail" in noise.inputs:
            noise.inputs["Detail"].default_value = float(rng.uniform(2.0, 8.0))
        ramp = tree.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = _rand_rgb(rng, sat=(0.1, 0.8), val=(0.05, 0.5))
        ramp.color_ramp.elements[1].color = _rand_rgb(rng, sat=(0.1, 0.8), val=(0.3, 0.95))
        tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        tree.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = float(rng.uniform(0.3, 1.2))
        kind = "noise"
    else:                                             # flat colour, as before
        bg.inputs["Color"].default_value = _rand_rgb(rng, sat=(0.0, 0.35), val=(0.05, 0.6))
        bg.inputs["Strength"].default_value = float(rng.uniform(0.2, 1.0))
        kind = "flat"
    return kind


def add_ground_plane(rng, drop=1.2):
    """A large ground plane below the object, present most of the time.

    Supplies a horizon, cast shadows and a receding textured surface -- the strongest
    single cue that separates a photograph from an object floating in void.
    """
    bpy.ops.mesh.primitive_plane_add(size=float(rng.uniform(14.0, 40.0)),
                                     location=(0.0, 0.0, -drop))
    plane = bpy.context.view_layer.objects.active
    plane.name = "ground"

    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.4, 1.0))

    if rng.random() < 0.65:                           # textured rather than flat
        tex = mat.node_tree.nodes.new(
            "ShaderNodeTexChecker" if rng.random() < 0.4 else "ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = float(rng.uniform(2.0, 25.0))
        if tex.type == "TEX_CHECKER":
            tex.inputs["Color1"].default_value = _rand_rgb(rng, sat=(0.0, 0.6))
            tex.inputs["Color2"].default_value = _rand_rgb(rng, sat=(0.0, 0.6))
            mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            ramp = mat.node_tree.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = _rand_rgb(rng, sat=(0.0, 0.7))
            ramp.color_ramp.elements[1].color = _rand_rgb(rng, sat=(0.0, 0.7))
            mat.node_tree.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
            mat.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = _rand_rgb(rng, sat=(0.0, 0.5))

    plane.data.materials.append(mat)
    return plane


def add_distractors(rng, max_n=4, min_radius=1.3, max_radius=3.6):
    """Scatter unrelated primitives around the target.

    These are the competing objects a real scene contains. They also produce partial
    occlusion, which the first dataset never showed the model at all.
    """
    made = []
    for i in range(int(rng.integers(0, max_n + 1))):
        op = getattr(bpy.ops.mesh, _DISTRACTOR_OPS[int(rng.integers(len(_DISTRACTOR_OPS)))])
        try:
            op()
        except Exception:
            continue
        obj = bpy.context.view_layer.objects.active
        obj.name = f"distractor_{i}"

        angle = rng.uniform(0.0, 6.283)
        radius = rng.uniform(min_radius, max_radius)
        obj.location = (radius * np.cos(angle) if False else float(radius * _cos(angle)),
                        float(radius * _sin(angle)),
                        float(rng.uniform(-0.9, 1.1)))
        obj.rotation_euler = tuple(float(v) for v in rng.uniform(0, 6.283, size=3))
        obj.scale = (float(rng.uniform(0.25, 0.95)),) * 3

        mat = bpy.data.materials.new(f"Distractor{i}")
        mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = _rand_rgb(rng, sat=(0.1, 0.9), val=(0.1, 0.9))
        bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.2, 1.0))
        obj.data.materials.append(mat)
        made.append(obj)
    return made


def set_clutter_visible(objects, visible):
    """Toggle render visibility -- used to isolate the target for its silhouette mask."""
    for obj in objects:
        obj.hide_render = not visible


def randomise_lighting(rng):
    """One sun plus 1-2 area lights at random positions, on top of the world lighting."""

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
