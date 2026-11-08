"""One-click cel / toon shading preset for imported ToaruIF models.

The game renders characters with ``Ter/Character/Toon*Ter``: two textures per
part (``_MainTex`` albedo and ``_MaskTex``) plus a handful of material
constants.  The mask is data, not colour, and the fragment shader only reads
its red and green channels:

* G (0.5 neutral) is added straight into the terminator: ``a + G - 1``.  1
  keeps a surface lit, 0 pins it in shadow for good -- which is exactly what
  the painted brow / lash / eye-pocket islands are for
* R and B are never sampled by any character variant (the eye highlight is
  painted into the albedo)

The shading is not taken from the scene lights.  ``_CharaLightDir`` is a fixed
character light direction; the vertex stage hands the fragment

    a   = clamp(dot(N_world, _CharaLightDir) + _LambertOffset, 0, 1)
    f   = clamp((clamp(a + G - 1, 0, 1) - 0.34) / 0.07, 0, 1)
    X   = albedo^4 + f * (albedo - albedo^4)
    out = albedo * (L*(1 + M*(f-1)) + A*(1 + M*(X-1)))

with ``M = _ToonShadeMask`` and ``L`` / ``A`` Unity's ``_LightColor0`` /
``_AmbientLightColor``.  Because the break sits on a world-space dot product it
stays put when the camera orbits, and the ``albedo^4`` term keeps the ambient
light on bright albedo -- white cloth does not go muddy in shadow.  This preset
reproduces those four lines directly; the earlier camera-space band could not
(its terminator followed the view, and a per-part tint replaced the two-light
blend).

The outline is an inverted hull built in geometry nodes, not a solidify shell.
The game shader draws it in a ``Cull Front`` pass whose vertex offset is
``(clipZ * _OutlineScaleTer + _OutlineOffsetTer) * 0.01`` -- after the
perspective divide that is a screen-constant part plus a world-space part over
depth.  The hull reproduces the screen-constant half with a depth-proportional
push (so the line keeps its width in *pixels* whatever the framing) and the
bold option adds a fixed world thickness on top, which is what makes close-ups
heavier exactly as in game.  The width is driven from the lens / sensor /
resolution; in the 3D view the reference point is the view origin instead of
the render camera, so scrolling in and out of the viewport keeps the same
width too.  Faces of eyes, brows, lashes and mouth are flagged out of the hull
by a face attribute.
"""

import math
import os

import bpy
from mathutils import Vector

TOON_KEY = "toaruif_toon_style"
MASK_SUFFIX = "_mask"
OUTLINE_MATERIAL = "ToaruIF-描边"
OUTLINE_MODIFIER = "ToaruIF-描边"
OUTLINE_GROUP = "ToaruIF-描边节点"
#: Bump whenever the node layout changes; the group is then rebuilt instead of
#: reused, which keeps .blend files made by an older version working.
GROUP_VERSION = 7
#: Per-face boolean written on the mesh: this face may grow an outline.
EDGE_ATTRIBUTE = "toaruif_edge"
#: Left behind by v0.5.0, which borrowed mmd_tools' edge preview.  Only used to
#: clean those files up when the preset is re-applied.
LEGACY_EDGE_MODIFIER = "mmd_edge_preview"
LEGACY_EDGE_PREFIX = "mmd_edge."

# Constants read out of the shipped character materials
# (model_base_004000901_* / 004001501_* in the asset bundles).  They are the
# defaults the game ships with, so a fresh preset lands on the same numbers.
TOON_SHADE_MASK = 0.35    # _ToonShadeMask -- weight of the light/ambient blend
LAMBERT_OFFSET = 0.40     # _LambertOffset -- where the break sits on the dot
#: The fragment shader's break: f = clamp((clamp(a + G - 1) - 0.34) / 0.07).
BREAK_THRESHOLD = 0.34
BREAK_WIDTH = 0.07
#: Horizontal angle of the character light, 0..1 (0.5 straight ahead); 0.42
#: leans it to the left, which is where the shipped character light sits.  The
#: game's ``_CharaLightDir`` is a full 3D direction, so an elevation is added
#: too -- the previous camera-space band dropped it entirely and could never
#: shade the top or the underside of anything.
LIGHT_DIRECTION = 0.42
LIGHT_ELEVATION = 25.0
#: Unity's ``_LightColor0`` / ``_AmbientLightColor``.  They sum to 1, so a fully
#: lit surface renders exactly as the albedo, the same as before.  Kept constant
#: on purpose: the preset must look identical whatever the scene lights are.
LIGHT_COLOR = (0.75, 0.75, 0.75)
AMBIENT_COLOR = (0.25, 0.25, 0.25)
RIM_COLOR = (1.0, 1.0, 1.0, 1.0)  # _RimColor
RIM_SCALE = 0.0           # _RimScale -- 0 in the assets; see the note below
RIM_OFFSET = 0.0          # _RimOffset -- shifts where the rim starts


def _srgb_to_linear(color):
    """A measured sRGB triple as linear scene values.

    Shader colour properties (emission, base colour, solid-view colour) are
    linear.  Feeding a screenshot value such as 0.208 straight in displays it
    as a bright olive, which is why the outline needs this conversion.
    """
    def channel(value):
        value = max(0.0, min(1.0, value))
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    linear = tuple(channel(c) for c in color[:3])
    return linear + (color[3] if len(color) > 3 else 1.0,)


# _OutlineColor is a warm brown, not black.  This is the value measured in the
# game (sRGB); the linear form below is what the materials actually store.
OUTLINE_COLOR_SRGB = (0.208, 0.114, 0.0, 1.0)
OUTLINE_COLOR = _srgb_to_linear(OUTLINE_COLOR_SRGB)
#: Materials that never grow an outline.  The PMX edge flag already covers
#: most models; these name hints catch the rest (eyes and brows painted onto
#: the face texture by the game's art team).
NO_EDGE_NAME_HINTS = ("瞳", "目", "眉", "睫", "口", "齿", "歯", "舌", "白目",
                      "eye", "brow", "lash", "iris", "pupil", "mouth",
                      "teeth", "tongue", "lip")

# Per part style.  body / hair / face are picked from the base texture name.
# The game has no per-part shading constants, so the only entry left is the
# strength of the optional mask-B highlight -- an extra no shader variant does.
STYLES = {
    "face": {"specular": 0.30},
    "hair": {"specular": 0.45},
    "body": {"specular": 0.20},
}

# Outline width in *rendered pixels*, not metres.  A solidify shell has a world
# space thickness, so one fixed number draws a hairline in a full-body shot and
# a thick band as soon as the camera moves in.  The modifier thickness is
# therefore driven from the camera distance and the focal terms, which keeps
# the line the same few pixels at any framing.
OUTLINE_PIXELS = 1.5
#: The "bold" option adds a fixed model-space thickness *on top of* the pixel
#: width, mirroring the game shader (its outline offset is ``clipZ *
#: _OutlineScaleTer + _OutlineOffsetTer``, i.e. a screen-constant part plus a
#: world-space part over depth).  The line never vanishes at distance and still
#: grows into a visibly heavier band as the camera moves in.
OUTLINE_BOLD_MM = 2.4


# ---------------------------------------------------------------------------
# material inspection
# ---------------------------------------------------------------------------

def base_image_of(material):
    """The albedo texture of a material (mask / toon / sphere maps excluded)."""
    if not material.use_nodes or material.node_tree is None:
        return None
    for node in material.node_tree.nodes:
        if node.bl_idname != "ShaderNodeTexImage" or node.image is None:
            continue
        name = node.image.name.lower()
        if (MASK_SUFFIX in name or "toon" in name or "_sph" in name
                or "_spa" in name or name.startswith("toaruif-")):
            continue
        return node.image
    return None


def part_of(name):
    lowered = (name or "").lower()
    if "face" in lowered:
        return "face"
    if "hair" in lowered:
        return "hair"
    return "body"


def _texture_dir(image):
    if image is None or not image.filepath:
        return None
    path = bpy.path.abspath(image.filepath)
    if not path or not os.path.isfile(path):
        return None
    return os.path.dirname(path)


def mask_image_for(image, part):
    """Find the ``_mask`` companion of an albedo texture.

    Hair ships under a different model id than its body, so the exact
    ``<stem>_mask`` name is tried first and then any ``*_<part>_mask`` in the
    same folder.
    """
    directory = _texture_dir(image)
    if directory is None:
        return None
    stem, ext = os.path.splitext(os.path.basename(image.filepath))
    exact = os.path.join(directory, stem + MASK_SUFFIX + ext)
    if os.path.isfile(exact):
        return _load(exact)
    for entry in sorted(os.listdir(directory)):
        lowered = entry.lower()
        if not lowered.endswith(MASK_SUFFIX + ext.lower()):
            continue
        if part in lowered:
            return _load(os.path.join(directory, entry))
    return None


def _load(path):
    image = bpy.data.images.load(path, check_existing=True)
    # The mask channels are data, not colour.  Doing this here, right after the
    # load, also avoids touching the colours space of an image that only exists
    # in memory: assigning a new one re-generates the buffer and would blank it.
    if image.colorspace_settings.name != "Non-Color":
        image.colorspace_settings.name = "Non-Color"
    return image


def ensure_default_mask():
    """A 4x4 image holding the neutral channel values (1.0, 0.5, 0.0)."""
    name = "ToaruIF-Mask默认"
    image = bpy.data.images.get(name)
    if image is not None:
        return image
    image = bpy.data.images.new(name, width=4, height=4, alpha=True)
    # Colours space first: changing it later clears a generated image in 5.x.
    image.colorspace_settings.name = "Non-Color"
    image.pixels = [1.0, 0.5, 0.0, 1.0] * 16
    image.use_fake_user = True
    return image


# ---------------------------------------------------------------------------
# node helpers
# ---------------------------------------------------------------------------

def _socket(node, collection, name, kind):
    for socket in collection:
        if socket.name == name and socket.type == kind:
            return socket
    raise KeyError("%s has no %s socket of type %s" % (node.bl_idname, name, kind))


def _new_mix(nodes, blend, factor=1.0):
    node = nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.blend_type = blend
    node.clamp_factor = True
    _socket(node, node.inputs, "Factor", "VALUE").default_value = factor
    return node


def _mix_color(nodes, blend, factor=1.0):
    node = _new_mix(nodes, blend, factor)
    a = _socket(node, node.inputs, "A", "RGBA")
    b = _socket(node, node.inputs, "B", "RGBA")
    result = _socket(node, node.outputs, "Result", "RGBA")
    return node, a, b, result


# ---------------------------------------------------------------------------
# shader construction
# ---------------------------------------------------------------------------

def build_toon_material(material, base_image, mask_image, style_key, style,
                        mask_strength=TOON_SHADE_MASK,
                        light_direction=LIGHT_DIRECTION,
                        break_width=BREAK_WIDTH,
                        lambert_offset=LAMBERT_OFFSET,
                        rim_scale=RIM_SCALE, mask_highlight=False):
    """Rebuild the material as the game's two-tone character shader.

    Every step follows the shipped fragment shader instead of an approximation:
    the break sits on a world-space dot product, the ``_mask`` green channel
    enters it at full weight, and the two light terms are blended with
    ``_ToonShadeMask`` around an ``albedo^4`` ambient guard.  ``break_width`` is
    the shader's 0.07 wide step and ``lambert_offset`` the ``_LambertOffset``
    added to the dot product.
    """
    tree = material.node_tree
    tree.nodes.clear()
    nodes = tree.nodes
    links = tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (1150, 0)
    emit = nodes.new("ShaderNodeEmission")
    emit.location = (950, 0)
    links.new(emit.outputs["Emission"], output.inputs["Surface"])

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (-1150, 300)
    tex.image = base_image

    mask_tex = nodes.new("ShaderNodeTexImage")
    mask_tex.location = (-1150, -350)
    mask_tex.image = mask_image

    separate = nodes.new("ShaderNodeSeparateColor")
    separate.location = (-900, -350)
    links.new(mask_tex.outputs["Color"], separate.inputs["Color"])

    # --- a = clamp(dot(N_world, _CharaLightDir) + _LambertOffset, 0, 1) -----
    # A fixed character light, not the scene's sun, and a full 3D dot: dropping
    # the vertical component (as the old camera-space band did) leaves the top
    # and the underside of every part shaded alike, and makes the terminator
    # follow the camera instead of staying on the model.
    azimuth = (light_direction - 0.5) * math.pi
    elevation = math.radians(LIGHT_ELEVATION)
    direction = (math.sin(azimuth) * math.cos(elevation),
                 -math.cos(azimuth) * math.cos(elevation),
                 math.sin(elevation))

    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-1150, 0)

    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.location = (-900, 0)
    dot.inputs[1].default_value = direction
    links.new(geometry.outputs["Normal"], dot.inputs[0])

    lambert = nodes.new("ShaderNodeMath")
    lambert.name = "toon_lit"
    lambert.location = (-700, 0)
    lambert.operation = "ADD"
    lambert.use_clamp = True
    links.new(dot.outputs["Value"], lambert.inputs[0])
    lambert.inputs[1].default_value = lambert_offset

    # --- f = clamp((clamp(a + G - 1, 0, 1) - 0.34) / 0.07, 0, 1) ----------
    # G enters at full weight: 1 keeps a texel lit, 0 pins it in shadow, so the
    # painted brow / lash islands stay dark whatever the light does.  Only
    # _ToonShadeMask scales the light/ambient blend below, never this break.
    plus_mask = nodes.new("ShaderNodeMath")
    plus_mask.location = (-520, 0)
    plus_mask.operation = "ADD"
    links.new(lambert.outputs[0], plus_mask.inputs[0])
    links.new(separate.outputs["Green"], plus_mask.inputs[1])

    minus_one = nodes.new("ShaderNodeMath")
    minus_one.location = (-360, 0)
    minus_one.operation = "SUBTRACT"
    minus_one.use_clamp = True
    minus_one.inputs[1].default_value = 1.0
    links.new(plus_mask.outputs[0], minus_one.inputs[0])

    threshold = nodes.new("ShaderNodeMath")
    threshold.location = (-200, 0)
    threshold.operation = "SUBTRACT"
    threshold.inputs[1].default_value = BREAK_THRESHOLD
    links.new(minus_one.outputs[0], threshold.inputs[0])

    lit = nodes.new("ShaderNodeMath")
    lit.name = "toon_lit_factor"
    lit.location = (-40, 0)
    lit.operation = "DIVIDE"
    lit.use_clamp = True
    lit.inputs[1].default_value = max(1e-3, break_width)
    links.new(threshold.outputs[0], lit.inputs[0])

    # --- X = albedo^4 + f * (albedo - albedo^4) ---------------------------
    # Bright albedo keeps almost all of its ambient term; that guard is what
    # stops white cloth, hats and scarves from going muddy in shadow.
    square = nodes.new("ShaderNodeVectorMath")
    square.operation = "MULTIPLY"
    square.location = (-700, 320)
    links.new(tex.outputs["Color"], square.inputs[0])
    links.new(tex.outputs["Color"], square.inputs[1])

    fourth = nodes.new("ShaderNodeVectorMath")
    fourth.operation = "MULTIPLY"
    fourth.location = (-520, 320)
    links.new(square.outputs["Vector"], fourth.inputs[0])
    links.new(square.outputs["Vector"], fourth.inputs[1])

    delta = nodes.new("ShaderNodeVectorMath")
    delta.operation = "SUBTRACT"
    delta.location = (-360, 320)
    links.new(tex.outputs["Color"], delta.inputs[0])
    links.new(fourth.outputs["Vector"], delta.inputs[1])

    weighed = nodes.new("ShaderNodeVectorMath")
    weighed.operation = "SCALE"
    weighed.location = (-200, 320)
    links.new(delta.outputs["Vector"], weighed.inputs[0])
    links.new(lit.outputs[0],
              _socket(weighed, weighed.inputs, "Scale", "VALUE"))

    spread = nodes.new("ShaderNodeVectorMath")
    spread.operation = "ADD"
    spread.location = (-40, 320)
    links.new(fourth.outputs["Vector"], spread.inputs[0])
    links.new(weighed.outputs["Vector"], spread.inputs[1])

    # --- albedo * (L*(1 + M*(f-1)) + A*(1 + M*(X-1))) ---------------------
    # Expanded to (L+A)*(1-M) + M*(L*f + A*X): one constant floor plus the two
    # masked light terms.  No per-part shadow tints -- the shader has none, the
    # only colour left in shadow is the light and ambient colour itself.
    weight = max(0.0, min(1.0, mask_strength))
    lit_gain = nodes.new("ShaderNodeVectorMath")
    lit_gain.operation = "SCALE"
    lit_gain.location = (150, 250)
    lit_gain.inputs[0].default_value = LIGHT_COLOR
    links.new(lit.outputs[0],
              _socket(lit_gain, lit_gain.inputs, "Scale", "VALUE"))

    ambient = nodes.new("ShaderNodeVectorMath")
    ambient.operation = "MULTIPLY"
    ambient.location = (150, 90)
    ambient.inputs[0].default_value = AMBIENT_COLOR
    links.new(spread.outputs["Vector"], ambient.inputs[1])

    blend = nodes.new("ShaderNodeVectorMath")
    blend.operation = "ADD"
    blend.location = (340, 170)
    links.new(lit_gain.outputs["Vector"], blend.inputs[0])
    links.new(ambient.outputs["Vector"], blend.inputs[1])

    masked = nodes.new("ShaderNodeVectorMath")
    masked.operation = "SCALE"
    masked.location = (530, 170)
    links.new(blend.outputs["Vector"], masked.inputs[0])
    _socket(masked, masked.inputs, "Scale", "VALUE").default_value = weight

    floor = nodes.new("ShaderNodeRGB")
    floor.location = (530, 330)
    floor.outputs["Color"].default_value = tuple(
        (light + ambient_light) * (1.0 - weight)
        for light, ambient_light in zip(LIGHT_COLOR, AMBIENT_COLOR)) + (1.0,)

    combined = nodes.new("ShaderNodeVectorMath")
    combined.operation = "ADD"
    combined.location = (720, 230)
    links.new(floor.outputs["Color"], combined.inputs[0])
    links.new(masked.outputs["Vector"], combined.inputs[1])

    shaded = nodes.new("ShaderNodeVectorMath")
    shaded.name = "toon_shaded"
    shaded.operation = "MULTIPLY"
    shaded.location = (900, 230)
    links.new(tex.outputs["Color"], shaded.inputs[0])
    links.new(combined.outputs["Vector"], shaded.inputs[1])

    # --- optional highlight from the mask's blue channel ------------------
    # No character variant samples the blue channel, so this is an extra the
    # user has to ask for; off by default to match the game.
    graded = shaded.outputs["Vector"]
    if mask_highlight:
        spec_scale = nodes.new("ShaderNodeMath")
        spec_scale.location = (900, -250)
        spec_scale.operation = "MULTIPLY"
        spec_scale.inputs[1].default_value = style["specular"]
        links.new(separate.outputs["Blue"], spec_scale.inputs[0])

        highlight, hi_a, hi_b, hi_result = _mix_color(nodes, "ADD")
        highlight.location = (1080, -80)
        links.new(shaded.outputs["Vector"], hi_a)
        hi_b.default_value = (1.0, 1.0, 1.0, 1.0)
        links.new(spec_scale.outputs[0],
                  _socket(highlight, highlight.inputs, "Factor", "VALUE"))
        graded = hi_result

    # --- rim: s = _RimScale * clamp((w + 0.5)^2 - 0.5, 0, 1) --------------
    # w = 1 - clamp(N_view.z - _RimOffset, 0, 1) in Unity's view space, where a
    # camera-facing normal is -Z; Blender's camera space flips that sign, hence
    # the negation.  The colour is base + s*(2*_RimColor - base), so s = 0.5
    # lands exactly on the rim colour.  _RimScale is 0 in the shipped
    # materials: the game only raises it at runtime through
    # InGameCharacterObject.SetRimSettings.
    view_normal = nodes.new("ShaderNodeVectorTransform")
    view_normal.vector_type = "NORMAL"
    view_normal.convert_from = "WORLD"
    view_normal.convert_to = "CAMERA"
    view_normal.location = (700, -520)
    links.new(geometry.outputs["Normal"], view_normal.inputs["Vector"])

    split = nodes.new("ShaderNodeSeparateXYZ")
    split.location = (880, -520)
    links.new(view_normal.outputs["Vector"], split.inputs["Vector"])

    away = nodes.new("ShaderNodeMath")
    away.location = (1040, -520)
    away.operation = "MULTIPLY"
    away.inputs[1].default_value = -1.0
    links.new(split.outputs["Z"], away.inputs[0])

    rim_start = nodes.new("ShaderNodeMath")
    rim_start.location = (1200, -520)
    rim_start.operation = "SUBTRACT"
    rim_start.use_clamp = True
    rim_start.inputs[1].default_value = RIM_OFFSET
    links.new(away.outputs[0], rim_start.inputs[0])

    rim_weight = nodes.new("ShaderNodeMath")
    rim_weight.location = (1360, -520)
    rim_weight.operation = "SUBTRACT"
    rim_weight.inputs[0].default_value = 1.0
    links.new(rim_start.outputs[0], rim_weight.inputs[1])

    rim_half = nodes.new("ShaderNodeMath")
    rim_half.location = (1520, -520)
    rim_half.operation = "ADD"
    rim_half.inputs[1].default_value = 0.5
    links.new(rim_weight.outputs[0], rim_half.inputs[0])

    rim_square = nodes.new("ShaderNodeMath")
    rim_square.location = (1680, -520)
    rim_square.operation = "MULTIPLY"
    links.new(rim_half.outputs[0], rim_square.inputs[0])
    links.new(rim_half.outputs[0], rim_square.inputs[1])

    rim_ramp = nodes.new("ShaderNodeMath")
    rim_ramp.location = (1840, -520)
    rim_ramp.operation = "SUBTRACT"
    rim_ramp.use_clamp = True
    rim_ramp.inputs[1].default_value = 0.5
    links.new(rim_square.outputs[0], rim_ramp.inputs[0])

    rim_gain = nodes.new("ShaderNodeMath")
    rim_gain.location = (2000, -520)
    rim_gain.operation = "MULTIPLY"
    rim_gain.inputs[1].default_value = max(0.0, rim_scale)
    links.new(rim_ramp.outputs[0], rim_gain.inputs[0])

    rim_color = nodes.new("ShaderNodeRGB")
    rim_color.location = (1840, -350)
    rim_color.outputs["Color"].default_value = RIM_COLOR

    rim_double = nodes.new("ShaderNodeVectorMath")
    rim_double.operation = "SCALE"
    rim_double.location = (2000, -350)
    links.new(rim_color.outputs["Color"], rim_double.inputs[0])
    _socket(rim_double, rim_double.inputs, "Scale", "VALUE").default_value = 2.0

    rim_delta = nodes.new("ShaderNodeVectorMath")
    rim_delta.operation = "SUBTRACT"
    rim_delta.location = (2170, -300)
    links.new(rim_double.outputs["Vector"], rim_delta.inputs[0])
    links.new(graded, rim_delta.inputs[1])

    rim_scaled = nodes.new("ShaderNodeVectorMath")
    rim_scaled.operation = "SCALE"
    rim_scaled.location = (2340, -300)
    links.new(rim_delta.outputs["Vector"], rim_scaled.inputs[0])
    links.new(rim_gain.outputs[0],
              _socket(rim_scaled, rim_scaled.inputs, "Scale", "VALUE"))

    rim_result = nodes.new("ShaderNodeVectorMath")
    rim_result.operation = "ADD"
    rim_result.location = (2510, -200)
    links.new(graded, rim_result.inputs[0])
    links.new(rim_scaled.outputs["Vector"], rim_result.inputs[1])

    links.new(rim_result.outputs["Vector"], emit.inputs["Color"])

    material[TOON_KEY] = style_key
    material.use_backface_culling = False


# ---------------------------------------------------------------------------
# outline
# ---------------------------------------------------------------------------

def outline_material(color=OUTLINE_COLOR):
    """The one inverted-hull material, shared by every outline face.

    The hull is pushed outward and its faces are flipped, so their normals point
    into the model.  Back-face culling is therefore *on*: the shell disappears
    over the surface and only the rim -- where those inverted faces turn edge-on
    -- is left over, which is exactly the silhouette line.
    """
    material = bpy.data.materials.get(OUTLINE_MATERIAL)
    if material is None:
        material = bpy.data.materials.new(OUTLINE_MATERIAL)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emit = tree.nodes.new("ShaderNodeEmission")
        tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
        material.use_fake_user = True
    # Eevee Next split this flag per pass; 4.2+ exposes the surface one separately.
    for attribute in ("use_backface_culling", "use_backface_culling_surface"):
        if hasattr(material, attribute):
            setattr(material, attribute, True)
    if material.node_tree is not None:
        for node in material.node_tree.nodes:
            if node.bl_idname == "ShaderNodeEmission":
                node.inputs["Color"].default_value = color
                break
    # Solid shading reads the viewport colour, not the node tree; without this
    # the outline shows up as light grey.
    material.diffuse_color = color
    return material


def _field_output(node, name, kind):
    """The socket of ``name``/``kind``; 3.6 lists one attribute output per type."""
    sockets = [socket for socket in node.outputs if socket.name == name]
    for socket in sockets:
        if socket.type == kind:
            return socket
    return sockets[0]


def _euler_of(nodes, links, rotation_socket):
    """Object Info reports an Euler in 3.6 and a Rotation from 4.x on."""
    if rotation_socket.type != "ROTATION":
        return rotation_socket
    convert = nodes.new("FunctionNodeRotationToEuler")
    links.new(rotation_socket, convert.inputs["Rotation"])
    return convert.outputs[0]


def outline_group():
    """The geometry-node hull, built once per .blend file.

    Every outline face is copied, the copy is pushed along its normal by
    ``k * distance to the camera`` and then flipped.  That is the whole trick:
    the projection divides by the same depth the offset multiplies by, so the
    rim keeps a constant *pixel* width at any framing.  A Solidify shell is a
    world-space thickness instead and can only be right at one depth, which is
    why a close-up used to turn the line into a fat, mushy band.

    The camera is fed in as a plain *position* that the modifier drives from
    ``scene.camera``; a captured Object pointer would go stale as soon as the
    scene camera is swapped, which is exactly when the width would silently
    drift again.  In the viewport the render camera is not what is being
    looked through -- scrolling dollies the view origin along the view axis --
    so there the depth is measured from the view matrix instead.  A final
    render reports ``Is Viewport`` as false and keeps using the camera.
    """
    group = bpy.data.node_groups.get(OUTLINE_GROUP)
    if group is not None and group.get("version") == GROUP_VERSION:
        return group
    if group is not None:
        bpy.data.node_groups.remove(group)
    tree = bpy.data.node_groups.new(OUTLINE_GROUP, "GeometryNodeTree")
    tree.use_fake_user = True
    nodes = tree.nodes
    links = tree.links

    def add_socket(name, in_out, kind):
        interface = getattr(tree, "interface", None)
        if interface is not None and hasattr(interface, "new_socket"):
            return interface.new_socket(name=name, in_out=in_out, socket_type=kind)
        # 3.6 still builds the interface from the two legacy collections.
        collection = tree.outputs if in_out == "OUTPUT" else tree.inputs
        return collection.new(kind, name)

    add_socket("Geometry", "INPUT", "NodeSocketGeometry")
    self_socket = add_socket("物体", "INPUT", "NodeSocketObject")
    camera_socket = add_socket("相机位置", "INPUT", "NodeSocketVector")
    material_socket = add_socket("描边材质", "INPUT", "NodeSocketMaterial")
    k_socket = add_socket("k", "INPUT", "NodeSocketFloat")
    pixels_socket = add_socket("像素恒定", "INPUT", "NodeSocketBool")
    width_socket = add_socket("世界宽度", "INPUT", "NodeSocketFloat")
    add_socket("Geometry", "OUTPUT", "NodeSocketGeometry")

    source = nodes.new("NodeGroupInput")
    sink = nodes.new("NodeGroupOutput")

    # --- which faces grow an outline -------------------------------------
    named = nodes.new("GeometryNodeInputNamedAttribute")
    if hasattr(named, "data_type"):
        try:
            named.data_type = "BOOLEAN"
        except TypeError:  # release without that hint
            pass
    named.inputs["Name"].default_value = EDGE_ATTRIBUTE
    duplicate = nodes.new("GeometryNodeDuplicateElements")
    duplicate.domain = "FACE"
    links.new(_field_output(named, "Attribute", "BOOLEAN"),
              duplicate.inputs["Selection"])
    links.new(source.outputs["Geometry"], duplicate.inputs["Geometry"])

    # Weld the copy back together and shade it smooth.  The PMX mesh is flat
    # shaded with split vertices, so each face carries its own normal; pushing
    # every face along its own normal tears the shell into separate quads whose
    # stray sides then show up as brown dashes across the hair.  Averaging the
    # normals over the welded copy keeps the shell one continuous surface, which
    # is the standard inverted-hull setup.
    welded = nodes.new("GeometryNodeMergeByDistance")
    welded.inputs["Distance"].default_value = 1e-4
    links.new(duplicate.outputs["Geometry"], welded.inputs[0])

    smooth = nodes.new("GeometryNodeSetShadeSmooth")
    smooth.inputs["Shade Smooth"].default_value = True
    links.new(welded.outputs["Geometry"], smooth.inputs[0])

    # --- depth: the local vertex has to be measured in world space --------
    # Geometry nodes run on local coordinates, so the camera distance is only
    # right once the object's own matrix (location, rotation and the 0.08
    # import scale) is applied.  Object Info carries that matrix live, so
    # moving or turning the model keeps the shell correct without re-applying
    # the preset.
    info = nodes.new("GeometryNodeObjectInfo")
    links.new(source.outputs["物体"], info.inputs["Object"])

    position = nodes.new("GeometryNodeInputPosition")
    normal = nodes.new("GeometryNodeInputNormal")

    scaled = nodes.new("ShaderNodeVectorMath")
    scaled.operation = "MULTIPLY"
    links.new(position.outputs["Position"], scaled.inputs[0])
    links.new(info.outputs["Scale"], scaled.inputs[1])

    turned = nodes.new("ShaderNodeVectorRotate")
    turned.rotation_type = "EULER_XYZ"
    links.new(scaled.outputs["Vector"], turned.inputs["Vector"])
    links.new(_euler_of(nodes, links, info.outputs["Rotation"]),
              turned.inputs["Rotation"])

    world = nodes.new("ShaderNodeVectorMath")
    world.operation = "ADD"
    links.new(turned.outputs["Vector"], world.inputs[0])
    links.new(info.outputs["Location"], world.inputs[1])

    distance = nodes.new("ShaderNodeVectorMath")
    distance.operation = "DISTANCE"
    links.new(world.outputs["Vector"], distance.inputs[0])
    links.new(source.outputs["相机位置"], distance.inputs[1])

    # --- the depth that is actually on screen -----------------------------
    # A render looks through ``scene.camera``, but the 3D view can be zoomed
    # anywhere: the scroll wheel dollies the view origin along the view axis
    # while the camera stays put, so a camera based depth makes the line grow
    # as the user zooms in -- the complaint this fixes.  Viewport Transform
    # reports the matrix of whatever is being drawn, so its inverse applied to
    # the origin is the point to measure depth from.  It is only meaningful in
    # the viewport, hence the switch; the minimum keeps a line that never turns
    # into a fat band even when a release hands back an empty matrix.
    depth_value = distance.outputs["Value"]
    try:
        viewport = nodes.new("GeometryNodeViewportTransform")
        in_viewport = nodes.new("GeometryNodeIsViewport")
        invert = nodes.new("FunctionNodeInvertMatrix")
        view_origin = nodes.new("ShaderNodeCombineXYZ")  # (0, 0, 0)
        to_world = nodes.new("FunctionNodeTransformPoint")
        view_distance = nodes.new("ShaderNodeVectorMath")
        nearest = nodes.new("ShaderNodeMath")
        choose = nodes.new("GeometryNodeSwitch")
        choose.input_type = "FLOAT"

        links.new(viewport.outputs["View"], invert.inputs["Matrix"])
        links.new(view_origin.outputs["Vector"], to_world.inputs["Vector"])
        links.new(invert.outputs["Matrix"], to_world.inputs["Transform"])
        view_distance.operation = "DISTANCE"
        links.new(world.outputs["Vector"], view_distance.inputs[0])
        links.new(to_world.outputs["Vector"], view_distance.inputs[1])
        nearest.operation = "MINIMUM"
        links.new(view_distance.outputs["Value"], nearest.inputs[0])
        links.new(distance.outputs["Value"], nearest.inputs[1])
        links.new(in_viewport.outputs["Is Viewport"], choose.inputs["Switch"])
        links.new(distance.outputs["Value"], choose.inputs["False"])  # render
        links.new(nearest.outputs[0], choose.inputs["True"])          # viewport
        depth_value = choose.outputs[0]
    except (RuntimeError, KeyError, AttributeError):
        pass  # 3.6 has no viewport transform node; the camera depth stands

    # --- width: k is "world units per unit of depth" ----------------------
    width = nodes.new("ShaderNodeMath")
    width.operation = "MULTIPLY"
    links.new(depth_value, width.inputs[0])
    links.new(source.outputs["k"], width.inputs[1])

    # Two width terms.  The pixel-constant part multiplies the depth by ``k``;
    # the game's outline offset is actually ``clipZ * _OutlineScaleTer +
    # _OutlineOffsetTer``, i.e. that screen-constant part *plus* a fixed world
    # thickness over depth.  The bold option therefore adds the world term on
    # top instead of replacing it -- the thin option keeps it at zero, and the
    # line never shrinks away at distance the way a pure world-space hull does.
    world = nodes.new("GeometryNodeSwitch")
    world.input_type = "FLOAT"
    links.new(source.outputs["像素恒定"], world.inputs["Switch"])
    links.new(source.outputs["世界宽度"], world.inputs["False"])
    world.inputs["True"].default_value = 0.0  # thin: pixels only

    sized = nodes.new("ShaderNodeMath")
    sized.operation = "ADD"
    links.new(width.outputs[0], sized.inputs[0])
    links.new(world.outputs[0], sized.inputs[1])

    # The push happens in local space, so the world width is divided by the
    # import scale (|scale| / sqrt(3) for the usual uniform 0.08).
    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    links.new(info.outputs["Scale"], length.inputs[0])
    unit = nodes.new("ShaderNodeMath")
    unit.operation = "DIVIDE"
    links.new(length.outputs["Value"], unit.inputs[0])
    unit.inputs[1].default_value = math.sqrt(3.0)
    local_width = nodes.new("ShaderNodeMath")
    local_width.operation = "DIVIDE"
    links.new(sized.outputs[0], local_width.inputs[0])
    links.new(unit.outputs[0], local_width.inputs[1])

    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "SCALE"
    links.new(normal.outputs["Normal"], offset.inputs[0])
    links.new(local_width.outputs[0], offset.inputs[3])

    moved = nodes.new("GeometryNodeSetPosition")
    links.new(smooth.outputs["Geometry"], moved.inputs["Geometry"])
    links.new(offset.outputs["Vector"], moved.inputs["Offset"])
    flipped = nodes.new("GeometryNodeFlipFaces")
    links.new(moved.outputs["Geometry"], flipped.inputs["Mesh"])
    painted = nodes.new("GeometryNodeSetMaterial")
    links.new(flipped.outputs["Mesh"], painted.inputs["Geometry"])
    links.new(source.outputs["描边材质"], painted.inputs["Material"])
    joined = nodes.new("GeometryNodeJoinGeometry")
    links.new(source.outputs["Geometry"], joined.inputs["Geometry"])
    links.new(painted.outputs["Geometry"], joined.inputs["Geometry"])
    links.new(joined.outputs["Geometry"], sink.inputs["Geometry"])

    tree["version"] = GROUP_VERSION
    tree["self_socket"] = self_socket.identifier
    tree["camera_socket"] = camera_socket.identifier
    tree["material_socket"] = material_socket.identifier
    tree["k_socket"] = k_socket.identifier
    tree["pixel_socket"] = pixels_socket.identifier
    tree["width_socket"] = width_socket.identifier
    return tree


def mark_edge_faces(obj):
    """Write the per-face "may grow an outline" flag that the hull reads."""
    mesh = obj.data
    attribute = mesh.attributes.get(EDGE_ATTRIBUTE)
    if attribute is None:
        attribute = mesh.attributes.new(EDGE_ATTRIBUTE, "BOOLEAN", "FACE")
    flags = [True if slot.material is None else not _no_edge_material(slot.material)
             for slot in obj.material_slots]
    values = []
    for poly in mesh.polygons:
        index = poly.material_index
        values.append(int(flags[index]) if 0 <= index < len(flags) else 1)
    attribute.data.foreach_set("value", values)
    return sum(values)


def _sensor_axis(data, rx, ry):
    """The render extent the sensor is fitted against, and its sensor size.

    ``sensor_fit`` decides which side of the sensor the lens value maps onto:
    AUTO puts ``sensor_width`` on the longer side of the image, HORIZONTAL pins
    it to the width and VERTICAL pins ``sensor_height`` to the height.  Using
    width / rx unconditionally draws a line 1.5x too thick on a portrait
    render, which is what made the preset camera's outline look heavy.
    """
    fit = getattr(data, "sensor_fit", "AUTO")
    if fit == "VERTICAL":
        return ry, data.sensor_height
    if fit == "HORIZONTAL":
        return rx, data.sensor_width
    if rx >= ry:
        return rx, data.sensor_width
    return ry, data.sensor_width


def pixel_k(pixels):
    """World units of push per unit of depth, for the current camera."""
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        return max(0.0, float(pixels)) * 0.001
    data = camera.data
    rx = max(1, scene.render.resolution_x)
    ry = max(1, scene.render.resolution_y)
    extent, sensor = _sensor_axis(data, rx, ry)
    return (max(0.0, float(pixels)) * max(sensor, 1e-6)
            / (max(data.lens, 1e-6) * extent))


def _input_paths(modifier, group, key):
    """Where this release keeps a node-group input (5.x moved it off ID props)."""
    identifier = group[key]
    paths = ['modifiers["%s"]["%s"]' % (modifier.name, identifier)]
    if hasattr(modifier, "properties"):
        paths.append('modifiers["%s"].properties.inputs.%s.value'
                     % (modifier.name, identifier))
    return paths


def set_input(modifier, group, key, value):
    """Write a group input on a modifier across the 3.6 / 5.x API split."""
    identifier = group[key]
    try:
        modifier[identifier] = value  # 3.6 - 4.x keep the inputs on ID properties
        return True
    except TypeError:
        pass
    try:  # 5.x: ``modifier.properties.inputs.<socket>.value``
        getattr(modifier.properties.inputs, identifier).value = value
        return True
    except Exception:  # noqa: BLE001 - socket not there
        return False


def _strip_drivers(obj, data_paths, indices):
    if obj.animation_data is None:
        return
    for data_path in data_paths:
        for index in indices:
            try:
                obj.driver_remove(data_path, index)
            except Exception:  # noqa: BLE001 - nothing driven there
                continue


def _scene_variable(driver, name, data_path):
    variable = driver.variables.new()
    variable.name = name
    variable.type = "SINGLE_PROP"
    variable.targets[0].id_type = "SCENE"
    variable.targets[0].id = bpy.context.scene
    variable.targets[0].data_path = data_path


def bind_width_driver(obj, modifier, group, pixels):
    """Make ``k`` follow the lens / sensor / resolution, so pixels stay pixels."""
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        return False
    paths = _input_paths(modifier, group, "k_socket")
    _strip_drivers(obj, paths, (-1,))
    driver = None
    for data_path in paths:
        try:
            driver = obj.driver_add(data_path).driver
            break
        except Exception:  # noqa: BLE001 - this release refuses that path
            continue
    if driver is None:
        return False
    driver.type = "SCRIPTED"
    for variable in list(driver.variables):
        driver.variables.remove(variable)

    _scene_variable(driver, "lens", "camera.data.lens")
    _scene_variable(driver, "sensor_w", "camera.data.sensor_width")
    _scene_variable(driver, "sensor_h", "camera.data.sensor_height")
    _scene_variable(driver, "rx", "render.resolution_x")
    _scene_variable(driver, "ry", "render.resolution_y")
    fit = getattr(camera.data, "sensor_fit", "AUTO")
    if fit == "VERTICAL":
        expression = "%g * sensor_h / (lens * ry)"
    elif fit == "HORIZONTAL":
        expression = "%g * sensor_w / (lens * rx)"
    else:
        expression = "%g * sensor_w / (lens * max(rx, ry))"
    driver.expression = expression % max(0.0, float(pixels))
    return driver.is_simple_expression


def bind_camera_position(obj, modifier, group):
    """Follow ``scene.camera`` through drivers, so a camera swap is harmless."""
    scene = bpy.context.scene
    camera = scene.camera
    paths = _input_paths(modifier, group, "camera_socket")
    _strip_drivers(obj, paths, (0, 1, 2))
    if camera is None:
        return False
    for axis in range(3):
        driver = None
        for data_path in paths:
            try:
                driver = obj.driver_add(data_path, axis).driver
                break
            except Exception:  # noqa: BLE001 - this release refuses that path
                continue
        if driver is None:
            return False
        driver.type = "SCRIPTED"
        for variable in list(driver.variables):
            driver.variables.remove(variable)
        _scene_variable(driver, "v", "camera.location[%d]" % axis)
        driver.expression = "v"
    return True


def add_outline(obj, pixels, color=OUTLINE_COLOR, bold=0.0):
    """Build the hull.  ``pixels`` is the screen-constant width, ``bold`` a
    world thickness added on top of it; 0 leaves the line pixel-only."""
    material = outline_material(color)
    faces = mark_edge_faces(obj)
    if faces == 0:
        return False
    slot_names = [slot.material.name for slot in obj.material_slots
                  if slot.material is not None]
    if material.name not in slot_names:
        obj.data.materials.append(material)
    modifier = obj.modifiers.get(OUTLINE_MODIFIER)
    if modifier is not None and modifier.type != "NODES":
        obj.modifiers.remove(modifier)
        modifier = None
    if modifier is None:
        modifier = obj.modifiers.new(OUTLINE_MODIFIER, "NODES")
    group = outline_group()
    modifier.node_group = group
    set_input(modifier, group, "self_socket", obj)
    camera = bpy.context.scene.camera
    if camera is not None:
        set_input(modifier, group, "camera_socket",
                  tuple(camera.matrix_world.translation))
    set_input(modifier, group, "material_socket", material)
    set_input(modifier, group, "k_socket", pixel_k(pixels))
    bold = max(0.0, float(bold))
    set_input(modifier, group, "pixel_socket", bold <= 0.0)
    set_input(modifier, group, "width_socket", bold)
    bind_width_driver(obj, modifier, group, pixels)
    bind_camera_position(obj, modifier, group)
    try:
        obj.update_tag()
    except Exception:  # noqa: BLE001 - not linked to a view layer yet
        pass
    return True


def remove_outline(obj):
    if obj.animation_data is not None:
        curves = obj.animation_data.drivers
        for curve in list(curves):
            if 'modifiers["%s"]' % OUTLINE_MODIFIER in curve.data_path:
                curves.remove(curve)
    for name in (OUTLINE_MODIFIER, LEGACY_EDGE_MODIFIER):
        modifier = obj.modifiers.get(name)
        if modifier is not None:
            obj.modifiers.remove(modifier)
    legacy = obj.vertex_groups.get(LEGACY_EDGE_MODIFIER)
    if legacy is not None:
        obj.vertex_groups.remove(legacy)
    attribute = obj.data.attributes.get(EDGE_ATTRIBUTE)
    if attribute is not None:
        obj.data.attributes.remove(attribute)
    material = bpy.data.materials.get(OUTLINE_MATERIAL)
    if material is None:
        return
    for index in range(len(obj.material_slots) - 1, -1, -1):
        if obj.material_slots[index].material == material:
            obj.data.materials.pop(index=index)


# --- mmd_tools edge preview ---------------------------------------------

def _mmd_edge_operator():
    try:
        operator = bpy.ops.mmd_tools.edge_preview_setup
        operator.get_rna_type()
        return operator
    except Exception:  # noqa: BLE001 - absent operator means "not available"
        return None


def _no_edge_material(material):
    """True when the material must not grow an outline."""
    name = (material.name or "").lower()
    for hint in NO_EDGE_NAME_HINTS:
        if hint in name:
            return True
    try:
        enabled = bool(getattr(material.mmd_material, "enabled_toon_edge", True))
    except Exception:  # noqa: BLE001 - material without mmd data draws an edge
        enabled = True
    return not enabled


def _activate(context, obj):
    try:
        context.view_layer.objects.active = obj
        obj.select_set(True)
        return True
    except Exception:  # noqa: BLE001 - object not in the view layer
        return False


def _clear_legacy_mmd(context, meshes):
    """Drop whatever v0.5.0 left behind: mmd_tools' edge preview shells."""
    operator = _mmd_edge_operator()
    if operator is not None and meshes and _activate(context, meshes[0]):
        try:
            operator(action="CLEAN")
        except Exception:  # noqa: BLE001 - manual cleanup below still runs
            pass
    for obj in meshes:
        modifier = obj.modifiers.get(LEGACY_EDGE_MODIFIER)
        if modifier is not None:
            obj.modifiers.remove(modifier)
        group = obj.vertex_groups.get(LEGACY_EDGE_MODIFIER)
        if group is not None:
            obj.vertex_groups.remove(group)
    for material in list(bpy.data.materials):
        if (material.name or "").startswith(LEGACY_EDGE_PREFIX):
            try:
                bpy.data.materials.remove(material)
            except Exception:  # noqa: BLE001 - still used by another scene
                pass


def setup_outline(context, meshes, width_px, color=OUTLINE_COLOR, bold=0.0):
    """Outlines as a hull in one of two width modes.

    ``bold`` > 0 adds a fixed world-space thickness to the pixel-constant
    width, which is how the game shader's outline behaves (heavier in
    close-ups); 0 keeps the line pixel-only.
    """
    clear_outline(context, meshes)
    built = [obj for obj in meshes if add_outline(obj, width_px, color, bold)]
    return "node" if built else "none"


def clear_outline(context, meshes):
    _clear_legacy_mmd(context, meshes)
    for obj in meshes:
        remove_outline(obj)


# ---------------------------------------------------------------------------
# scene setup
# ---------------------------------------------------------------------------

def ensure_engine(scene):
    """Eevee, under whichever id it goes by in this Blender release."""
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = name
        except TypeError:  # not an engine of this version
            continue
        return name
    return scene.render.engine


def ensure_view_transform(scene):
    """Standard instead of AgX/Filmic: a cel ramp should not be tone mapped."""
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("ToaruIF-环境")
        scene.world = world
    world.use_nodes = True
    for node in world.node_tree.nodes:
        if node.bl_idname == "ShaderNodeBackground":
            node.inputs["Color"].default_value = (0.051, 0.051, 0.051, 1.0)
            node.inputs["Strength"].default_value = 1.0
            break


def ensure_light(scene):
    """A single sun from the upper left for the scene.

    The cel shading runs entirely through Emission (the band is camera-space),
    so this light no longer drives the character; it is there for anything else
    in the scene.  Shadow casting stays off because Eevee resolves it with a
    dithered penumbra, which reads as speckle next to a hard cel break.
    """
    for obj in scene.objects:
        if obj.type == "LIGHT":
            return obj, False
    data = bpy.data.lights.new("ToaruIF-主光", "SUN")
    data.energy = 3.0
    data.use_shadow = False
    obj = bpy.data.objects.new("ToaruIF-主光", data)
    scene.collection.objects.link(obj)
    obj.rotation_euler = (math.radians(55.0), 0.0, math.radians(35.0))
    return obj, True


def ensure_camera(scene, meshes):
    """A full-body front camera, only when the scene has none yet."""
    existing = [obj for obj in scene.objects if obj.type == "CAMERA"]
    if existing:
        scene.camera = existing[0]
        return existing[0], False

    xs, ys, zs = [], [], []
    for obj in meshes:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            xs.append(world_corner.x)
            ys.append(world_corner.y)
            zs.append(world_corner.z)
    if not xs:
        return None, False

    center = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
    height = max(max(zs) - min(zs), 1e-3)
    distance = height * 2.0
    data = bpy.data.cameras.new("ToaruIF-相机")
    data.lens = 50.0
    data.sensor_fit = "VERTICAL"
    data.sensor_height = 24.0
    obj = bpy.data.objects.new("ToaruIF-相机", data)
    scene.collection.objects.link(obj)
    obj.location = (center[0], center[1] - distance, center[2])
    obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    scene.camera = obj
    return obj, True
