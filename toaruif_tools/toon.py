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

The outline also follows BV-Toon: mmd_tools' own ``edge_preview_setup`` builds
one inverted-hull modifier per mesh plus a ``mmd_edge.<material>`` shell material
per slot, so every material keeps its own colour.  Materials that must not grow
an edge (eyes, brows, lashes, mouth) get their shell vertex weights zeroed out.
"""

import math
import os

import bpy
from mathutils import Vector

TOON_KEY = "toaruif_toon_style"
MASK_SUFFIX = "_mask"
OUTLINE_MATERIAL = "ToaruIF-描边"
OUTLINE_MODIFIER = "ToaruIF-描边"
EDGE_MODIFIER = "mmd_edge_preview"
EDGE_MATERIAL_PREFIX = "mmd_edge."
#: mmd_tools bakes this weight into ``mmd_edge_preview``; a vertex weighted 1.0
#: therefore only gets 2% of the modifier thickness as real shell.
EDGE_WEIGHT_FACTOR = 0.02

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
# _OutlineColor is a warm brown, not black.  _OutlineScaleTer is 0.05 and
# _OutlineOffsetTer 0.1, but those are shader-space factors, so the metre
# value below stays hand tuned.
OUTLINE_COLOR = (0.208, 0.114, 0.0, 1.0)
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

# Outline width in Blender units (metres) measured on the finished shell.
# 1.8 mm reads as a thin anime line; above ~3 mm starts swallowing fingers.
OUTLINE_WIDTH = 0.0018


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
    """Fallback hull material, used only when mmd_tools is not available."""
    material = bpy.data.materials.get(OUTLINE_MATERIAL)
    if material is None:
        material = bpy.data.materials.new(OUTLINE_MATERIAL)
        material.use_nodes = True
        material.use_backface_culling = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emit = tree.nodes.new("ShaderNodeEmission")
        tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
        material.use_fake_user = True
    if material.node_tree is not None:
        for node in material.node_tree.nodes:
            if node.bl_idname == "ShaderNodeEmission":
                node.inputs["Color"].default_value = color
                break
    # Solid shading reads the viewport colour, not the node tree; without this
    # the hull shows up as light grey instead of an outline.
    material.diffuse_color = color
    return material


def add_hull_outline(obj, thickness, color=OUTLINE_COLOR):
    """Inverted-hull outline through Solidify's flipped normals.

    ``use_even_offset`` stays off on purpose: it normalises the offset by the
    angle to the neighbouring face, so the near-degenerate triangles around the
    fingers get a huge displacement and shoot long black splinters out of the
    hand.
    """
    material = outline_material(color)
    slots = list(obj.material_slots)
    if not slots or slots[0].material is None:
        return False
    index = len(slots)
    obj.data.materials.append(material)
    modifier = obj.modifiers.get(OUTLINE_MODIFIER)
    if modifier is None:
        modifier = obj.modifiers.new(OUTLINE_MODIFIER, "SOLIDIFY")
    modifier.thickness = thickness
    modifier.offset = 1.0
    modifier.use_flip_normals = True
    modifier.use_rim = False
    modifier.use_even_offset = False
    modifier.material_offset = index
    return True


def remove_hull_outline(obj):
    modifier = obj.modifiers.get(OUTLINE_MODIFIER)
    if modifier is not None:
        obj.modifiers.remove(modifier)
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


def apply_edge_settings(meshes, width, color=OUTLINE_COLOR):
    """Match mmd_tools' shell to the game's thin warm-brown line.

    mmd_tools stores ``edge_weight * 0.02`` per vertex, so the modifier
    thickness must be divided by that factor to get ``width`` metres of shell.
    The edge material feeds its colour through a Gamma 2.2 node, hence the
    ``** 2.2`` below: what the user picks is what the render shows.
    """
    thickness = max(0.0, float(width)) / EDGE_WEIGHT_FACTOR
    for obj in meshes:
        modifier = obj.modifiers.get(EDGE_MODIFIER)
        if modifier is None:
            continue
        scale = obj.matrix_world.to_scale()
        factor = max((abs(scale.x) + abs(scale.y) + abs(scale.z)) / 3.0, 1e-6)
        modifier.thickness = thickness / factor
        modifier.thickness_vertex_group = 1e-3
        modifier.use_even_offset = False

    write = tuple(min(1.0, max(0.0, float(value))) ** 2.2 for value in color[:3])
    seen = set()
    for obj in meshes:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.name in seen:
                continue
            if not material.name.startswith(EDGE_MATERIAL_PREFIX):
                continue
            seen.add(material.name)
            if material.name == EDGE_MATERIAL_PREFIX + "disabled":
                continue
            material.diffuse_color = (float(color[0]), float(color[1]),
                                      float(color[2]), 1.0)
            if not material.use_nodes or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if (node.bl_idname == "ShaderNodeGroup"
                        and node.node_tree is not None
                        and node.node_tree.name == "MMDEdgePreview"):
                    for socket in node.inputs:
                        if socket.type == "RGBA" and not socket.links:
                            socket.default_value = (write[0], write[1], write[2], 1.0)


def drop_disabled_edges(meshes):
    """Zero the shell weight on vertices only surrounded by no-edge faces.

    mmd_tools builds the shell for *every* vertex and merely covers the ones
    with ``edge_flag = 0`` with a transparent material, so eyes and mouth still
    get a full-thickness hull.  Clearing those weights removes the geometry.
    Vertices shared with a drawn material keep their weight so the boundary is
    not eaten away.
    """
    dropped = 0
    for obj in meshes:
        modifier = obj.modifiers.get(EDGE_MODIFIER)
        if modifier is None:
            continue
        name = getattr(modifier, "vertex_group", "")
        group = obj.vertex_groups.get(name) if name else None
        if group is None:
            continue
        flags = []
        for slot in obj.material_slots:
            material = slot.material
            flags.append(True if material is None else not _no_edge_material(material))
        if all(flags):
            continue
        keep = {}
        for poly in obj.data.polygons:
            index = poly.material_index
            wanted = flags[index] if 0 <= index < len(flags) else True
            for vertex in poly.vertices:
                keep[vertex] = keep.get(vertex, False) or wanted
        victims = [vertex for vertex, wanted in keep.items() if not wanted]
        if victims:
            group.add(victims, 0.0, "REPLACE")
            dropped += len(victims)
    return dropped


def add_mmd_outline(context, meshes, width, color=OUTLINE_COLOR):
    operator = _mmd_edge_operator()
    if operator is None or not meshes:
        return False
    if not _activate(context, meshes[0]):
        return False
    try:
        operator(action="CREATE")
    except Exception:  # noqa: BLE001 - let the hull fallback take over
        return False
    apply_edge_settings(meshes, width, color)
    drop_disabled_edges(meshes)
    return True


def remove_mmd_outline(context, meshes):
    operator = _mmd_edge_operator()
    if operator is not None and meshes and _activate(context, meshes[0]):
        try:
            operator(action="CLEAN")
        except Exception:  # noqa: BLE001 - manual cleanup below still runs
            pass
    for obj in meshes:
        modifier = obj.modifiers.get(EDGE_MODIFIER)
        if modifier is not None:
            obj.modifiers.remove(modifier)
        group = obj.vertex_groups.get(EDGE_MODIFIER)
        if group is not None:
            obj.vertex_groups.remove(group)
    for material in list(bpy.data.materials):
        if (material.name or "").startswith(EDGE_MATERIAL_PREFIX):
            try:
                bpy.data.materials.remove(material)
            except Exception:  # noqa: BLE001 - still used by another scene
                pass


def setup_outline(context, meshes, width, color=OUTLINE_COLOR):
    """Outlines through mmd_tools' edge preview, local hull as the fallback."""
    clear_outline(context, meshes)
    if add_mmd_outline(context, meshes, width, color):
        return "mmd"
    for obj in meshes:
        add_hull_outline(obj, width, color)
    return "hull"


def clear_outline(context, meshes):
    remove_mmd_outline(context, meshes)
    for obj in meshes:
        remove_hull_outline(obj)


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
