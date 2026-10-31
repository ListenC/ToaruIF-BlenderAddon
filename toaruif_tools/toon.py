"""One-click cel / toon shading preset for imported ToaruIF models.

The game renders characters with a cel shader fed by three textures per part:
the base colour map plus a ``_mask`` map whose channels are data, not colour.
Sampling the shipped masks shows the layout:

* R stays at ~1.0 everywhere (a weight, not a picture)
* G sits at 0.5 almost everywhere, with islands of 0 (face: eye/brow/lash
  shapes, body: small pockets, hair: 19% of the texels) and 1.0 in hair
* B is 0 except a few percent on the face (eye highlight)
* A is fully opaque

The 4x4 ``Orange.png`` shipped next to the textures is exactly (1.0, 0.5, 0.0)
-- the neutral value of those three channels -- so the intended default is
"R=1, G=0.5, B=0".  That maps cleanly onto a cel shader:

* G = how much the surface takes shading (0.5 neutral, 0 = self-lit, 1 = extra
  dark)
* B = highlight amount
* R = overall shadow weight

The shipped character materials confirm that much: they bind exactly two
textures (``_MainTex`` + ``_MaskTex``) and the shader's property list is
``_MainTex, _MaskTex, _LambertOffset, _ToonShadeMask, _RimColor, _RimScale,
_RimOffset, _OutlineScaleTer, _OutlineOffsetTer, _OutlineZBiasTer,
_OutlineColor, _Alpha`` -- the face variant adds ``_EyebrowsZShiftRate`` and
the projected-light one adds ``_ProjLightTex / Scale / WorldScale``.  There is
no ramp texture, so the break is computed from the light response plus those
constants.  Everything else on the material (``_Color``, ``_AmbientLightColor``,
``_Mode``, ``_Glossiness`` ...) is a Standard-shader leftover and is absent
from that list, which is why it is ignored here as well.

The light/shadow break is not driven by the scene lights.  Following BV-Toon's
approach, a *vertical band* is built from the camera-space normal: the normal's
horizontal components are dotted with a light direction that can be rotated
with one number, and a narrow soft step turns that into a hard cel terminator.
That makes the look deterministic -- rotating the view does not turn the
character into a different time of day -- and the ``_ToonShadeMask`` still moves
the break per texel through the mask's green channel.

The outline is an inverted hull built in geometry nodes, not a solidify shell.
Each face that may grow an edge is duplicated, welded smooth, pushed along its
normal by a distance proportional to its depth from the camera and then flipped;
the projection divides by the same depth, so the line keeps a constant width in
*pixels* whatever the framing.  The width is driven from the lens / sensor /
resolution, so a close-up no longer turns the line into a fat band.  Faces of
eyes, brows, lashes and mouth are flagged out of the hull by a face attribute.
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
GROUP_VERSION = 4
#: Per-face boolean written on the mesh: this face may grow an outline.
EDGE_ATTRIBUTE = "toaruif_edge"
#: Left behind by v0.5.0, which borrowed mmd_tools' edge preview.  Only used to
#: clean those files up when the preset is re-applied.
LEGACY_EDGE_MODIFIER = "mmd_edge_preview"
LEGACY_EDGE_PREFIX = "mmd_edge."

# Constants read out of the shipped character materials
# (model_base_004000901_* / 004001501_* in the asset bundles).  They are the
# defaults the game ships with, so a fresh preset lands on the same numbers.
TOON_SHADE_MASK = 0.35    # _ToonShadeMask -- weight of the mask green channel
#: BV-Toon's preset values.  The break is near-hard (0.04 on a -1..1 band is a
#: couple of degrees) and the light comes from the upper left.
LIGHT_DIRECTION = 0.42
BREAK_WIDTH = 0.04
#: Band value the terminator sits on.  ``_LambertOffset`` (0.4) is an offset on
#: a dot product in the game, while the band above is a camera-space normal
#: component -- different quantities, so it cannot be reused verbatim.  BV-Toon
#: cuts at 0.5 and a front facing model then only darkens near the silhouette
#: (a thin rim, no visible two-tone). Parked near 0 the terminator runs through
#: the open surface, which is what makes the cel look read at a glance.
BREAK_CENTRE = 0.05
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
# The shadow is the base texture pushed toward ``shadow_tint``; that tint is a
# hand tuned darkening, the game itself has no shadow-colour parameter.
STYLES = {
    "face": {
        "shadow_tint": (0.72, 0.44, 0.44, 1.0),
        "specular": 0.30,
    },
    "hair": {
        "shadow_tint": (0.40, 0.30, 0.52, 1.0),
        "specular": 0.45,
    },
    "body": {
        "shadow_tint": (0.32, 0.28, 0.52, 1.0),
        "specular": 0.20,
    },
}

# Outline width in *rendered pixels*, not metres.  A solidify shell has a world
# space thickness, so one fixed number draws a hairline in a full-body shot and
# a thick band as soon as the camera moves in.  The modifier thickness is
# therefore driven from the camera distance and the focal terms, which keeps
# the line the same few pixels at any framing.
OUTLINE_PIXELS = 1.5


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
    return bpy.data.images.load(path, check_existing=True)


def ensure_default_mask():
    """A 4x4 image holding the neutral channel values (1.0, 0.5, 0.0)."""
    name = "ToaruIF-Mask默认"
    image = bpy.data.images.get(name)
    if image is not None:
        return image
    image = bpy.data.images.new(name, width=4, height=4, alpha=True)
    image.pixels = [1.0, 0.5, 0.0, 1.0] * 16
    image.colorspace_settings.name = "Non-Color"
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
                        break_centre=BREAK_CENTRE, rim_scale=RIM_SCALE):
    """Rebuild the material as the game's two-tone character shader.

    The terminator comes from a camera-space vertical band (BV-Toon), so it
    follows ``light_direction`` instead of the scene lights.  ``mask_strength``
    is ``_ToonShadeMask``: the green channel of ``_mask`` moves that break per
    texel -- 0 (painted shadow islands) darkens, 1 (always-lit) brightens and
    0.5 is neutral.
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
    if mask_image.colorspace_settings.name != "Non-Color":
        mask_image.colorspace_settings.name = "Non-Color"

    separate = nodes.new("ShaderNodeSeparateColor")
    separate.location = (-900, -350)
    links.new(mask_tex.outputs["Color"], separate.inputs["Color"])

    # --- camera-space vertical band --------------------------------------
    # BV-Toon's band: the camera-space normal's horizontal component *only*
    # (the up/down one is dropped, so the band stays vertical however the
    # camera is tilted), scaled by the light direction.  Surfaces whose
    # horizontal normal points away from the light land in shadow -- a true
    # vertical terminator, not a Fresnel ring.
    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-1150, 0)

    to_camera = nodes.new("ShaderNodeVectorTransform")
    to_camera.vector_type = "NORMAL"
    to_camera.convert_from = "WORLD"
    to_camera.convert_to = "CAMERA"
    to_camera.location = (-950, 0)

    split = nodes.new("ShaderNodeSeparateXYZ")
    split.location = (-750, 0)

    angle = (light_direction - 0.5) * math.pi * 2.0
    band = nodes.new("ShaderNodeMath")
    band.name = "toon_band"
    band.location = (-550, 0)
    band.operation = "MULTIPLY"
    band.inputs[1].default_value = math.cos(angle)
    links.new(split.outputs["X"], band.inputs[0])

    links.new(geometry.outputs["Normal"], to_camera.inputs["Vector"])
    links.new(to_camera.outputs["Vector"], split.inputs["Vector"])

    # G moves the break: 0.5 is neutral, 0 keeps the painted islands (brows,
    # lashes, body pockets) lit and 1 pushes the hair / cloth strands darker.
    bias = nodes.new("ShaderNodeMath")
    bias.location = (-350, -300)
    bias.operation = "SUBTRACT"
    bias.inputs[1].default_value = 0.5
    links.new(separate.outputs["Green"], bias.inputs[0])

    shift = nodes.new("ShaderNodeMath")
    shift.location = (-150, -300)
    shift.operation = "MULTIPLY"
    shift.inputs[1].default_value = 2.0 * max(0.0, min(1.0, mask_strength))
    links.new(bias.outputs[0], shift.inputs[0])

    broken = nodes.new("ShaderNodeMath")
    broken.location = (30, 0)
    broken.operation = "ADD"
    links.new(band.outputs[0], broken.inputs[0])
    links.new(shift.outputs[0], broken.inputs[1])

    # shadow = clamp((band - (break_centre - width/2)) / width): a near-hard
    # terminator parked inside the visible surface, nudged per texel by the mask.
    centre = break_centre - max(1e-3, break_width) * 0.5
    step = nodes.new("ShaderNodeMath")
    step.location = (260, 0)
    step.operation = "SUBTRACT"
    links.new(broken.outputs[0], step.inputs[0])
    step.inputs[1].default_value = centre

    shadow = nodes.new("ShaderNodeMath")
    shadow.name = "toon_shadow"
    shadow.location = (440, 0)
    shadow.operation = "DIVIDE"
    shadow.use_clamp = True
    shadow.inputs[1].default_value = max(1e-3, break_width)
    links.new(step.outputs[0], shadow.inputs[0])

    # --- colours ---------------------------------------------------------
    shade_node, shade_a, shade_b, shade_result = _mix_color(nodes, "MULTIPLY", 1.0)
    shade_node.location = (300, -160)
    tint = nodes.new("ShaderNodeRGB")
    tint.location = (80, -230)
    tint.outputs["Color"].default_value = style["shadow_tint"]
    links.new(tex.outputs["Color"], shade_a)
    links.new(tint.outputs["Color"], shade_b)

    final_node, final_a, final_b, final_result = _mix_color(nodes, "MIX")
    final_node.location = (640, 0)
    links.new(tex.outputs["Color"], final_a)
    links.new(shade_result, final_b)
    links.new(shadow.outputs[0],
              _socket(final_node, final_node.inputs, "Factor", "VALUE"))

    # --- highlight from the mask's blue channel --------------------------
    spec_scale = nodes.new("ShaderNodeMath")
    spec_scale.location = (460, -300)
    spec_scale.operation = "MULTIPLY"
    spec_scale.inputs[1].default_value = style["specular"]
    links.new(separate.outputs["Blue"], spec_scale.inputs[0])

    highlight, hi_a, hi_b, hi_result = _mix_color(nodes, "ADD")
    highlight.location = (760, -60)
    links.new(final_result, hi_a)
    hi_b.default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(spec_scale.outputs[0], _socket(highlight, highlight.inputs, "Factor", "VALUE"))

    # --- rim light -------------------------------------------------------
    # _RimScale is 0 in the shipped materials: the game only raises it at
    # runtime (InGameCharacterObject.SetRimSettings, tweened through a
    # MaterialPropertyBlock), so it ships here as an explicit strength and
    # stays off at its default.
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.location = (300, -480)
    fresnel.inputs["IOR"].default_value = 1.45

    rim_ramp = nodes.new("ShaderNodeMapRange")
    rim_ramp.location = (480, -480)
    rim_ramp.interpolation_type = "SMOOTHSTEP"
    rim_ramp.clamp = True
    rim_ramp.inputs["From Min"].default_value = RIM_OFFSET
    rim_ramp.inputs["From Max"].default_value = 1.0
    rim_ramp.inputs["To Min"].default_value = 0.0
    rim_ramp.inputs["To Max"].default_value = 1.0
    links.new(fresnel.outputs["Fac"], rim_ramp.inputs["Value"])

    rim_gain = nodes.new("ShaderNodeMath")
    rim_gain.location = (660, -480)
    rim_gain.operation = "MULTIPLY"
    rim_gain.inputs[1].default_value = max(0.0, rim_scale)
    links.new(rim_ramp.outputs["Result"], rim_gain.inputs[0])

    rim_add, rim_a, rim_b, rim_result = _mix_color(nodes, "ADD")
    rim_add.location = (880, -80)
    links.new(hi_result, rim_a)
    rim_b.default_value = RIM_COLOR
    links.new(rim_gain.outputs[0],
              _socket(rim_add, rim_add.inputs, "Factor", "VALUE"))

    links.new(rim_result, emit.inputs["Color"])

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
    drift again.
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

    # --- width: k is "world units per unit of depth" ----------------------
    width = nodes.new("ShaderNodeMath")
    width.operation = "MULTIPLY"
    links.new(distance.outputs["Value"], width.inputs[0])
    links.new(source.outputs["k"], width.inputs[1])

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
    links.new(width.outputs[0], local_width.inputs[0])
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


def add_outline(obj, pixels, color=OUTLINE_COLOR):
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


def setup_outline(context, meshes, width_px, color=OUTLINE_COLOR):
    """Outlines as a depth-proportional hull, so the width stays in pixels."""
    clear_outline(context, meshes)
    built = [obj for obj in meshes if add_outline(obj, width_px, color)]
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
