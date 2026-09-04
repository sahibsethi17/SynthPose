"""Procedural object library for the synthetic dataset.

Every shape here is built from Blender primitives at render time, so the dataset
reproduces from a seed alone with no downloaded assets and no licensing questions.

**Why these three shapes.** The obvious starter set (cube, sphere) is unusable for
pose regression: a sphere has no recoverable orientation at all, and a cube maps to
the same image under 24 distinct rotations. Training a single-mode regressor against
such a target is ill-posed -- the optimal prediction is the mean of the equivalent
rotations, so the loss plateaus in a way that reads like a bug but is really the
label being ambiguous. Each shape below has a trivial rotational symmetry group,
so image -> rotation is a genuine function.
"""

from __future__ import annotations

import bpy

#: Object categories, in the label's class-index order.
CATEGORIES = ("suzanne", "mug", "bracket")


def _activate(obj):
    """Make ``obj`` the sole selected+active object (required by bpy.ops)."""
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def _join(objs):
    """Join ``objs`` into the first one and return it."""
    _activate(objs[0])
    for o in objs[1:]:
        o.select_set(True)
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _canonicalise(obj):
    """Centre the origin on the geometry and scale so the longest side is 1.

    Both steps matter for the labels. Centring makes the translation target mean
    the object's centre rather than an arbitrary primitive origin, and unit-sizing
    decouples "how big is this shape" from the per-sample scale we sample later,
    so apparent size in the image is driven by distance and scale alone.
    """
    _activate(obj)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    longest = max(obj.dimensions)
    obj.scale = (1.0 / longest,) * 3
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_mode = "QUATERNION"
    return obj


def _build_suzanne(rng):
    """Blender's monkey head: strongly asymmetric on all three axes."""
    bpy.ops.mesh.primitive_monkey_add(size=2.0)
    obj = bpy.context.view_layer.objects.active
    if rng.random() < 0.5:                      # vary silhouette smoothness
        mod = obj.modifiers.new("Subsurf", "SUBSURF")
        mod.levels = mod.render_levels = 1
    return obj


def _build_mug(rng):
    """A tapered, open-topped cup with a handle offset above its mid-height.

    Every detail here exists to kill a symmetry. The first version was a *closed,
    centred cylinder* with the handle at mid-height, which is exactly invariant under a
    180 deg rotation about the handle axis -- flip it upside down and the shape is
    unchanged. That is a true 2-fold rotational symmetry, and the trained model duly put
    36% of its mug predictions ~180 deg from the truth.

    Three independent fixes, so no single one has to carry it:
      * the body tapers (a real cup is wider at the rim), breaking top/bottom symmetry;
      * the rim is open and the base solid, so the two ends look different;
      * the handle sits above mid-height rather than centred on it.
    """
    r_base = rng.uniform(0.32, 0.45)
    r_rim = r_base * rng.uniform(1.15, 1.40)          # taper: rim wider than base
    height = rng.uniform(0.85, 1.25)

    bpy.ops.mesh.primitive_cone_add(radius1=r_base, radius2=r_rim,
                                    depth=height, vertices=48)
    body = bpy.context.view_layer.objects.active

    # Hollow the cup: a slightly smaller inverted frustum boolean-subtracted from the
    # top leaves an open rim and a solid base, which is what distinguishes the ends.
    wall = rng.uniform(0.05, 0.09)
    bpy.ops.mesh.primitive_cone_add(radius1=max(r_base - wall, 0.05), radius2=r_rim - wall,
                                    depth=height, vertices=48,
                                    location=(0.0, 0.0, wall * 2.0))
    cavity = bpy.context.view_layer.objects.active

    _activate(body)
    mod = body.modifiers.new("Hollow", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cavity
    bpy.ops.object.modifier_apply(modifier="Hollow")
    bpy.data.objects.remove(cavity, do_unlink=True)

    handle_r = rng.uniform(0.17, 0.26)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=handle_r,
        minor_radius=handle_r * rng.uniform(0.22, 0.32),
        major_segments=32,
        minor_segments=12,
        location=(r_rim * 0.9 + handle_r * 0.5, 0.0, height * rng.uniform(0.05, 0.15)),
        rotation=(1.5707963, 0.0, 0.0),
    )
    handle = bpy.context.view_layer.objects.active
    return _join([body, handle])


def _build_bracket(rng):
    """An L-bracket, thickened and with a raised boss on one face.

    The first version was near-planar and mirror-symmetric about its own mid-plane. A
    flat object with two identical faces looks the same flipped front-to-back -- the
    projection cannot say which face you are seeing -- and the model put 19.5% of its
    bracket predictions ~180 deg out. Being *near*-symmetric rather than exactly
    symmetric made it harder to spot than the mug, not easier to learn.

    The boss is the load-bearing fix: it sits on one face only, so no rotation maps the
    shape to itself. Thicker arms and unequal cross-sections make the two faces
    additionally distinguishable under perspective.
    """
    a = float(rng.uniform(0.80, 1.25))          # long arm
    b = float(rng.uniform(0.55, 0.90))          # upright arm
    t = float(rng.uniform(0.26, 0.38))          # thicker than the flat original

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    arm_x = bpy.context.view_layer.objects.active
    arm_x.scale = (a, t, t)
    arm_x.location = (a / 2.0, 0.0, 0.0)

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    arm_z = bpy.context.view_layer.objects.active
    arm_z.scale = (t, t * rng.uniform(0.62, 0.78), b)   # unequal cross-section
    arm_z.location = (t / 2.0, t * rng.uniform(0.10, 0.20), b / 2.0)   # offset off-plane

    # Boss on the +Y face only -- this is what removes the mirror symmetry outright.
    boss_r = t * rng.uniform(0.30, 0.42)
    boss_d = t * rng.uniform(0.55, 0.85)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=boss_r, depth=boss_d, vertices=20,
        location=(a * rng.uniform(0.55, 0.78), t / 2.0 + boss_d / 2.0, 0.0),
        rotation=(1.5707963, 0.0, 0.0),
    )
    boss = bpy.context.view_layer.objects.active

    for o in (arm_x, arm_z, boss):
        _activate(o)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return _join([arm_x, arm_z, boss])


_BUILDERS = {
    "suzanne": _build_suzanne,
    "mug": _build_mug,
    "bracket": _build_bracket,
}


def build(category: str, rng):
    """Create a canonicalised object of ``category`` at the world origin."""
    obj = _BUILDERS[category](rng)
    obj.name = f"target_{category}"
    return _canonicalise(obj)
