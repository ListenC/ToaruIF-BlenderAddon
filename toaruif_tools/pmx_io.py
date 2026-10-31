"""Minimal PMX 2.0 / 2.1 reader and writer.

Derived from the workspace toolkit ``pmx_kit/pmxkit/pmx_io.py`` and trimmed to
what the importer needs: parse a whole file, edit the model dict, write it back
with every untouched byte reproduced.  The one addition over the original is
the PMX 2.1 SoftBody section (it sits after the joints and is simply dropped
when the model is written as 2.0).

Design notes
------------
* Structures are plain dicts keyed like the on-disk record, so a transform is a
  dictionary edit.
* Every variable-width field is decided by the values that were read, never by
  guesses, so a load/dump round trip reproduces the input byte for byte.
* A few exporters pad material / bone morph offsets with extra bytes the spec
  does not describe.  Those are preserved verbatim and the width is
  auto-detected per file.
"""

import struct

__all__ = ["PmxModel", "PmxError", "load", "loads", "save", "dumps",
           "MORPH_TYPES", "WEIGHT_TYPES"]

SIGNATURE = b"PMX "

# PMX spec vertex weight types. QDEF (4) is a 2.1 extension; its payload layout
# is identical to BDEF4, so a 2.0 file can carry it as BDEF4 after a downgrade.
WEIGHT_TYPES = {0: "BDEF1", 1: "BDEF2", 2: "BDEF4", 3: "SDEF", 4: "QDEF"}

MORPH_TYPES = {
    0: "group", 1: "vertex", 2: "bone", 3: "uv1", 4: "uv2", 5: "uv3",
    6: "uv4", 7: "uv5", 8: "material", 9: "flip", 10: "impulse",
}

# Extra bytes some exporters append to a morph offset, beyond the documented
# layout.  The bytes are kept verbatim per offset and these are the widths tried
# in order when auto-detecting.  0 comes first everywhere because that is the
# spec.
_MATERIAL_PAD_CANDIDATES = (0, 8, 4, 12, 16, 6, 2, 10, 14, 20)
_BONE_PAD_CANDIDATES = (0, 16, 4, 12, 8)


class PmxError(Exception):
    """Raised for malformed input or an impossible write request."""


# ---------------------------------------------------------------------------
# binary primitives
# ---------------------------------------------------------------------------

class _Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def raw(self, count):
        end = self.pos + count
        if end > len(self.data):
            raise PmxError("truncated at %d: want %d bytes, have %d"
                           % (self.pos, count, len(self.data) - self.pos))
        chunk = self.data[self.pos:end]
        self.pos = end
        return chunk

    def u8(self):
        return self.raw(1)[0]

    def u16(self):
        return struct.unpack_from("<H", self.raw(2))[0]

    def i32(self):
        return struct.unpack_from("<i", self.raw(4))[0]

    def f32(self):
        return struct.unpack_from("<f", self.raw(4))[0]

    def vec(self, count):
        return list(struct.unpack_from("<%df" % count, self.raw(4 * count)))

    def sint(self, size):
        """Signed index of ``size`` bytes; -1 means none."""
        if size == 1:
            return struct.unpack_from("<b", self.raw(1))[0]
        if size == 2:
            return struct.unpack_from("<h", self.raw(2))[0]
        return self.i32()

    def uint(self, size):
        """Unsigned reference index (vertex / morph / rigid body); -1 means none."""
        if size == 1:
            value = self.u8()
            return None if value == 0xFF else value
        if size == 2:
            value = self.u16()
            return None if value == 0xFFFF else value
        value = self.i32()
        return None if value == -1 else value

    def text(self, encoding):
        length = self.i32()
        if length == 0:
            return ""
        if length < 0:
            # Some writers store a character count as a negative number.
            return self.raw(-length * 2).decode(encoding, "replace")
        return self.raw(length).decode(encoding, "replace")


class _Writer:
    __slots__ = ("buf",)

    def __init__(self):
        self.buf = bytearray()

    def raw(self, chunk):
        self.buf += chunk

    def u8(self, value):
        self.buf.append(value & 0xFF)

    def u16(self, value):
        self.buf += struct.pack("<H", value & 0xFFFF)

    def i32(self, value):
        self.buf += struct.pack("<i", value)

    def f32(self, value):
        self.buf += struct.pack("<f", value)

    def vec(self, values):
        self.buf += struct.pack("<%df" % len(values), *values)

    def sint(self, size, value):
        if size == 1:
            self.buf.append((0xFF if value is None else value) & 0xFF)
        elif size == 2:
            self.buf += struct.pack("<H", (0xFFFF if value is None else value) & 0xFFFF)
        else:
            self.buf += struct.pack("<i", -1 if value is None else value)

    def uint(self, size, value):
        if value is None:
            value = 0xFF if size == 1 else (0xFFFF if size == 2 else -1)
        if size == 1:
            self.buf.append(value & 0xFF)
        elif size == 2:
            self.buf += struct.pack("<H", value & 0xFFFF)
        else:
            self.buf += struct.pack("<i", value)

    def text(self, encoding, value):
        encoded = value.encode(encoding) if value else b""
        self.i32(len(encoded))
        self.buf += encoded


# ---------------------------------------------------------------------------
# model container
# ---------------------------------------------------------------------------

class PmxModel:
    """One PMX file, section by section.

    ``globals`` holds the raw header bytes after the version (encoding, extra-UV
    count and the six index sizes) so an unusual file survives a round trip.
    """

    def __init__(self):
        self.version = 2.0
        self.globals = [0, 0, 2, 2, 2, 2, 2, 2]
        self.name = ""
        self.name_en = ""
        self.comment = ""
        self.comment_en = ""
        self.vertices = []
        self.faces = []
        self.faces_signed = False
        self.textures = []
        self.materials = []
        self.bones = []
        self.morphs = []
        self.display_frames = []
        self.rigid_bodies = []
        self.joints = []
        # PMX 2.1 only; dropped when written as 2.0.
        self.soft_bodies = []
        # Non-spec trailing bytes per morph offset, in bytes; 0 for spec files.
        self.material_morph_pad = 0
        self.bone_morph_pad = 0

    @property
    def encoding(self):
        return "utf-16-le" if self.globals[0] == 0 else "utf-8"

    @property
    def add_uv_count(self):
        return self.globals[1]

    @property
    def sizes(self):
        """(vertex, texture, material, bone, morph, rigid body) index widths."""
        return tuple(self.globals[2:8])

    def morph_names(self):
        return [morph["name"] for morph in self.morphs]

    def summary(self):
        return {
            "version": self.version,
            "name": self.name,
            "vertices": len(self.vertices),
            "faces": len(self.faces),
            "textures": len(self.textures),
            "materials": len(self.materials),
            "bones": len(self.bones),
            "morphs": len(self.morphs),
            "display_frames": len(self.display_frames),
            "rigid_bodies": len(self.rigid_bodies),
            "joints": len(self.joints),
            "soft_bodies": len(self.soft_bodies),
        }


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def loads(data, material_pad=None, bone_pad=None):
    """Parse PMX bytes into a :class:`PmxModel`.

    ``material_pad`` / ``bone_pad`` pin the morph-offset extras instead of
    auto-detecting them.
    """
    if material_pad is not None and bone_pad is not None:
        model, consumed = _parse(data, material_pad, bone_pad,
                                 want_consumed=True)
        if consumed != len(data):
            raise PmxError("trailing data: %d of %d bytes consumed"
                           % (consumed, len(data)))
        return model

    short_model = None
    consumed = None
    last = None
    try:
        short_model, consumed = _parse(data, 0, 0, want_consumed=True)
    except PmxError as exc:
        last = exc
    if short_model is not None and consumed == len(data):
        return short_model

    for candidate in _pad_candidates(short_model, data, consumed):
        try:
            model, offset = _parse(data, candidate[0], candidate[1],
                                   want_consumed=True)
        except PmxError as exc:
            last = exc
            continue
        if offset == len(data):
            model.material_morph_pad = candidate[0]
            model.bone_morph_pad = candidate[1]
            return model
        last = "consumed %d of %d bytes (material morph pad=%d, " \
               "bone morph pad=%d)" % (offset, len(data),
                                       candidate[0], candidate[1])
    raise PmxError("cannot parse PMX: %s" % last)


def _pad_candidates(model, data, consumed):
    """Pad widths to try, best guess first, without duplicates."""
    ordered = []
    if model is not None and consumed is not None:
        material_count = sum(len(morph["offsets"]) for morph in model.morphs
                             if morph["type"] == 8)
        bone_count = sum(len(morph["offsets"]) for morph in model.morphs
                         if morph["type"] == 2)
        shortfall = len(data) - consumed
        for bone_width in _BONE_PAD_CANDIDATES:
            for material_width in _MATERIAL_PAD_CANDIDATES:
                if material_width or bone_width:
                    if material_width * material_count \
                            + bone_width * bone_count == shortfall:
                        ordered.append((material_width, bone_width))
    ordered.extend((material_width, bone_width)
                   for bone_width in _BONE_PAD_CANDIDATES
                   for material_width in _MATERIAL_PAD_CANDIDATES
                   if material_width or bone_width)
    seen = set()
    for candidate in ordered:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def load(path):
    with open(path, "rb") as handle:
        return loads(handle.read())


def _parse(data, material_pad, bone_pad, want_consumed=False):
    reader = _Reader(data)
    signature = reader.raw(4)
    if signature != SIGNATURE:
        raise PmxError("not a PMX file: signature is %r" % signature)

    model = PmxModel()
    model.material_morph_pad = material_pad
    model.bone_morph_pad = bone_pad
    model.version = reader.f32()
    count = reader.u8()
    model.globals = list(reader.raw(count))
    encoding = model.encoding
    vertex_size, texture_size, material_size, bone_size, morph_size, rigid_size \
        = model.sizes

    model.name = reader.text(encoding)
    model.name_en = reader.text(encoding)
    model.comment = reader.text(encoding)
    model.comment_en = reader.text(encoding)

    # -- vertices ---------------------------------------------------------
    for _ in range(reader.i32()):
        vertex = {
            "pos": reader.vec(3),
            "normal": reader.vec(3),
            "uv": reader.vec(2),
            "add_uv": reader.vec(4 * model.add_uv_count)
            if model.add_uv_count else [],
        }
        weight_type = reader.u8()
        vertex["weight_type"] = weight_type
        if weight_type == 0:
            vertex["weight"] = {"bones": [reader.sint(bone_size)]}
        elif weight_type == 1:
            vertex["weight"] = {"bones": [reader.sint(bone_size),
                                          reader.sint(bone_size)],
                                "weights": [reader.f32()]}
        elif weight_type == 2 or weight_type == 4:
            vertex["weight"] = {"bones": [reader.sint(bone_size)
                                          for _ in range(4)],
                                "weights": [reader.f32() for _ in range(4)]}
        elif weight_type == 3:
            vertex["weight"] = {"bones": [reader.sint(bone_size),
                                          reader.sint(bone_size)],
                                "weights": [reader.f32()],
                                "c": reader.vec(3),
                                "r0": reader.vec(3),
                                "r1": reader.vec(3)}
        else:
            raise PmxError("vertex %d: unknown weight type %d"
                           % (len(model.vertices), weight_type))
        vertex["edge_scale"] = reader.f32()
        model.vertices.append(vertex)

    # -- faces ------------------------------------------------------------
    # The count is documented as positive; a negative one (seen in exports that
    # mean "at most this many") is remembered so the byte stream survives.
    face_count = reader.i32()
    model.faces_signed = face_count < 0
    count = abs(face_count)
    if vertex_size == 1:
        model.faces = list(reader.raw(count))
    elif vertex_size == 2:
        model.faces = list(struct.unpack_from("<%dH" % count,
                                              reader.raw(2 * count)))
    else:
        model.faces = list(struct.unpack_from("<%di" % count,
                                              reader.raw(4 * count)))

    # -- textures ---------------------------------------------------------
    for _ in range(reader.i32()):
        model.textures.append(reader.text(encoding))

    # -- materials --------------------------------------------------------
    for _ in range(reader.i32()):
        material = {
            "name": reader.text(encoding),
            "name_en": reader.text(encoding),
            "diffuse": reader.vec(4),
            "specular": reader.vec(3),
            "shininess": reader.f32(),
            "ambient": reader.vec(3),
            "flag": reader.u8(),
            "edge_color": reader.vec(4),
            "edge_size": reader.f32(),
            "tex": reader.sint(texture_size),
            "env": reader.sint(texture_size),
            "env_mode": reader.u8(),
            "toon_shared": reader.u8(),
        }
        material["toon"] = reader.u8() if material["toon_shared"] == 1 \
            else reader.sint(texture_size)
        material["memo"] = reader.text(encoding)
        material["surface_count"] = reader.i32()
        model.materials.append(material)

    # -- bones ------------------------------------------------------------
    for _ in range(reader.i32()):
        bone = {"name": reader.text(encoding), "name_en": reader.text(encoding),
                "pos": reader.vec(3), "parent": reader.sint(bone_size),
                "layer": reader.i32()}
        flag = reader.u16()
        bone["flag"] = flag
        if flag & 0x0001:
            bone["tail_bone"] = reader.sint(bone_size)
            bone["tail_offset"] = None
        else:
            bone["tail_bone"] = None
            bone["tail_offset"] = reader.vec(3)
        if flag & 0x0300:                       # 0x0100 rotate / 0x0200 move
            bone["inherit_parent"] = reader.sint(bone_size)
            bone["inherit_weight"] = reader.f32()
        else:
            bone["inherit_parent"] = None
            bone["inherit_weight"] = None
        bone["fixed_axis"] = reader.vec(3) if flag & 0x0400 else None
        if flag & 0x0800:
            bone["local_x"] = reader.vec(3)
            bone["local_z"] = reader.vec(3)
        else:
            bone["local_x"] = bone["local_z"] = None
        bone["ext_key"] = reader.i32() if flag & 0x1000 else None
        if flag & 0x0020:
            ik = {"target": reader.sint(bone_size), "loop": reader.i32(),
                  "limit": reader.f32(), "links": []}
            for _ in range(reader.i32()):
                link = {"bone": reader.sint(bone_size), "limit": reader.u8()}
                if link["limit"]:
                    link["lo"] = reader.vec(3)
                    link["hi"] = reader.vec(3)
                ik["links"].append(link)
            bone["ik"] = ik
        else:
            bone["ik"] = None
        model.bones.append(bone)

    # -- morphs -----------------------------------------------------------
    for _ in range(reader.i32()):
        morph = {"name": reader.text(encoding), "name_en": reader.text(encoding),
                 "panel": reader.u8(), "type": reader.u8(), "offsets": []}
        count = reader.i32()
        kind = morph["type"]
        for _ in range(count):
            if kind == 0 or kind == 9:           # group / flip
                morph["offsets"].append({"index": reader.sint(morph_size),
                                         "weight": reader.f32()})
            elif kind == 1:                      # vertex
                morph["offsets"].append({"index": reader.uint(vertex_size),
                                         "pos": reader.vec(3)})
            elif kind == 2:                      # bone
                offset = {"index": reader.sint(bone_size),
                          "pos": reader.vec(3),
                          "rot": reader.vec(4)}
                if bone_pad:
                    offset["pad"] = reader.raw(bone_pad)
                morph["offsets"].append(offset)
            elif 3 <= kind <= 7:                 # uv1..uv5
                morph["offsets"].append({"index": reader.uint(vertex_size),
                                         "uv": reader.vec(4)})
            elif kind == 8:                      # material
                morph["offsets"].append({
                    "index": reader.sint(material_size),
                    "op": reader.u8(),
                    "diffuse": reader.vec(4),
                    "specular": reader.vec(3),
                    "shininess": reader.f32(),
                    "ambient": reader.vec(3),
                    "edge_color": reader.vec(4),
                    "edge_size": reader.f32(),
                    "tex_coef": reader.vec(4),
                    "env_coef": reader.vec(4),
                    "toon_coef": reader.vec(4),
                })
                if material_pad:
                    morph["offsets"][-1]["pad"] = reader.raw(material_pad)
            elif kind == 10:                     # impulse
                morph["offsets"].append({"index": reader.sint(rigid_size),
                                         "local": reader.u8(),
                                         "velocity": reader.vec(3),
                                         "torque": reader.vec(3)})
            else:
                raise PmxError("morph %r: unknown type %d"
                               % (morph["name"], kind))
        morph["offset_count"] = count
        model.morphs.append(morph)

    # -- display frames ---------------------------------------------------
    for _ in range(reader.i32()):
        frame = {"name": reader.text(encoding), "name_en": reader.text(encoding),
                 "special": reader.u8(), "items": []}
        for _ in range(reader.i32()):
            kind = reader.u8()
            index = reader.sint(bone_size if kind == 0 else morph_size)
            frame["items"].append([kind, index])
        model.display_frames.append(frame)

    # -- rigid bodies -----------------------------------------------------
    for _ in range(reader.i32()):
        model.rigid_bodies.append({
            "name": reader.text(encoding),
            "name_en": reader.text(encoding),
            "bone": reader.sint(bone_size),
            "group": reader.u8(),
            "no_collide": reader.u16(),
            "shape": reader.u8(),
            "size": reader.vec(3),
            "pos": reader.vec(3),
            "rot": reader.vec(3),
            "mass": reader.f32(),
            "lin_damp": reader.f32(),
            "ang_damp": reader.f32(),
            "restitution": reader.f32(),
            "friction": reader.f32(),
            "mode": reader.u8(),
        })

    # -- joints -----------------------------------------------------------
    for _ in range(reader.i32()):
        model.joints.append({
            "name": reader.text(encoding),
            "name_en": reader.text(encoding),
            "type": reader.u8(),
            "a": reader.sint(rigid_size),
            "b": reader.sint(rigid_size),
            "pos": reader.vec(3),
            "rot": reader.vec(3),
            "pos_lo": reader.vec(3),
            "pos_hi": reader.vec(3),
            "rot_lo": reader.vec(3),
            "rot_hi": reader.vec(3),
            "spring_pos": reader.vec(3),
            "spring_rot": reader.vec(3),
        })

    # -- soft bodies (PMX 2.1, appended after the joints) ------------------
    if reader.pos < len(data):
        for _ in range(reader.i32()):
            body = {
                "name": reader.text(encoding),
                "name_en": reader.text(encoding),
                "shape": reader.u8(),
                "material": reader.sint(material_size),
                "group": reader.u8(),
                "no_collide": reader.u16(),
                "flag": reader.u8(),
                "b_link_distance": reader.i32(),
                "cluster_count": reader.i32(),
                "total_mass": reader.f32(),
                "collision_margin": reader.f32(),
                "aero_model": reader.i32(),
                "config": reader.vec(12),
                "cluster": reader.vec(6),
                "iteration": [reader.i32() for _ in range(4)],
                "material_param": reader.vec(3),
                "anchors": [],
                "pins": [],
            }
            for _ in range(reader.i32()):
                body["anchors"].append({
                    "rigid": reader.sint(rigid_size),
                    "vertex": reader.uint(vertex_size),
                    "near": reader.u8(),
                })
            for _ in range(reader.i32()):
                body["pins"].append(reader.uint(vertex_size))
            model.soft_bodies.append(body)

    if want_consumed:
        return model, reader.pos
    if reader.pos != len(data):
        raise PmxError("trailing data: %d of %d bytes consumed"
                       % (reader.pos, len(data)))
    return model


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _write_pad(writer, offset, width):
    """Append a morph offset's non-spec trailing bytes, exactly ``width`` long."""
    if not width:
        return
    pad = offset.get("pad")
    if pad is None:
        writer.raw(b"\x00" * width)
    elif len(pad) == width:
        writer.raw(pad)
    else:
        writer.raw((pad + b"\x00" * width)[:width])


def required_sizes(model):
    """The narrowest index width each reference class actually needs.

    The header travels with the file, so a model that grows past a boundary
    keeps its old width and every index above the boundary overflows; widening
    is therefore what keeps the file readable by a strict reader.
    """
    counts = (len(model.vertices), len(model.textures), len(model.materials),
              len(model.bones), len(model.morphs), len(model.rigid_bodies))
    out = []
    for count in counts:
        if count > 0x7FFF:
            out.append(4)
        elif count > 0x7F:
            out.append(2)
        else:
            out.append(1)
    return out


def dumps(model):
    """Serialise a :class:`PmxModel` back to PMX bytes (always as 2.0 layout)."""
    writer = _Writer()
    globals_ = list(model.globals)
    if len(globals_) < 8:
        raise PmxError("header globals must hold at least 8 bytes")

    # Never write an index wider than the header promises, and raise the promise
    # when the model has outgrown it.
    wanted = required_sizes(model)
    for slot in range(6):
        current = globals_[2 + slot]
        if current not in (1, 2, 4):
            raise PmxError("index width %d in the header is not 1, 2 or 4"
                           % current)
        if wanted[slot] > current:
            globals_[2 + slot] = wanted[slot]

    writer.raw(SIGNATURE)
    writer.f32(model.version)
    writer.u8(len(globals_))
    writer.raw(bytes(globals_))

    encoding = model.encoding
    vertex_size, texture_size, material_size, bone_size, morph_size, rigid_size \
        = tuple(globals_[2:8])

    writer.text(encoding, model.name)
    writer.text(encoding, model.name_en)
    writer.text(encoding, model.comment)
    writer.text(encoding, model.comment_en)

    writer.i32(len(model.vertices))
    for vertex in model.vertices:
        writer.vec(vertex["pos"])
        writer.vec(vertex["normal"])
        writer.vec(vertex["uv"])
        if model.add_uv_count:
            writer.vec(vertex["add_uv"])
        weight_type = vertex["weight_type"]
        weight = vertex["weight"]
        writer.u8(weight_type)
        if weight_type == 0:
            writer.sint(bone_size, weight["bones"][0])
        elif weight_type == 1:
            writer.sint(bone_size, weight["bones"][0])
            writer.sint(bone_size, weight["bones"][1])
            writer.f32(weight["weights"][0])
        elif weight_type == 2 or weight_type == 4:
            for index in range(4):
                writer.sint(bone_size, weight["bones"][index])
            for index in range(4):
                writer.f32(weight["weights"][index])
        elif weight_type == 3:
            writer.sint(bone_size, weight["bones"][0])
            writer.sint(bone_size, weight["bones"][1])
            writer.f32(weight["weights"][0])
            writer.vec(weight["c"])
            writer.vec(weight["r0"])
            writer.vec(weight["r1"])
        else:
            raise PmxError("unknown vertex weight type %d" % weight_type)
        writer.f32(vertex["edge_scale"])

    writer.i32(-len(model.faces) if model.faces_signed else len(model.faces))
    for index in model.faces:
        if vertex_size == 1:
            writer.u8(index)
        elif vertex_size == 2:
            writer.u16(index)
        else:
            writer.i32(index)

    writer.i32(len(model.textures))
    for texture in model.textures:
        writer.text(encoding, texture)

    writer.i32(len(model.materials))
    for material in model.materials:
        writer.text(encoding, material["name"])
        writer.text(encoding, material["name_en"])
        writer.vec(material["diffuse"])
        writer.vec(material["specular"])
        writer.f32(material["shininess"])
        writer.vec(material["ambient"])
        writer.u8(material["flag"])
        writer.vec(material["edge_color"])
        writer.f32(material["edge_size"])
        writer.sint(texture_size, material["tex"])
        writer.sint(texture_size, material["env"])
        writer.u8(material["env_mode"])
        writer.u8(material["toon_shared"])
        if material["toon_shared"] == 1:
            writer.u8(material["toon"])
        else:
            writer.sint(texture_size, material["toon"])
        writer.text(encoding, material["memo"])
        writer.i32(material["surface_count"])

    writer.i32(len(model.bones))
    for bone in model.bones:
        writer.text(encoding, bone["name"])
        writer.text(encoding, bone["name_en"])
        writer.vec(bone["pos"])
        writer.sint(bone_size, bone["parent"])
        writer.i32(bone["layer"])
        flag = bone["flag"]
        writer.u16(flag)
        if flag & 0x0001:
            writer.sint(bone_size, bone["tail_bone"])
        else:
            writer.vec(bone["tail_offset"])
        if flag & 0x0300:
            writer.sint(bone_size, bone["inherit_parent"])
            writer.f32(bone["inherit_weight"])
        if flag & 0x0400:
            writer.vec(bone["fixed_axis"])
        if flag & 0x0800:
            writer.vec(bone["local_x"])
            writer.vec(bone["local_z"])
        if flag & 0x1000:
            writer.i32(bone["ext_key"])
        if flag & 0x0020:
            ik = bone["ik"]
            writer.sint(bone_size, ik["target"])
            writer.i32(ik["loop"])
            writer.f32(ik["limit"])
            writer.i32(len(ik["links"]))
            for link in ik["links"]:
                writer.sint(bone_size, link["bone"])
                writer.u8(link["limit"])
                if link["limit"]:
                    writer.vec(link["lo"])
                    writer.vec(link["hi"])

    writer.i32(len(model.morphs))
    for morph in model.morphs:
        writer.text(encoding, morph["name"])
        writer.text(encoding, morph["name_en"])
        writer.u8(morph["panel"])
        writer.u8(morph["type"])
        kind = morph["type"]
        offsets = morph["offsets"]
        writer.i32(len(offsets))
        for offset in offsets:
            if kind == 0 or kind == 9:
                writer.sint(morph_size, offset["index"])
                writer.f32(offset["weight"])
            elif kind == 1:
                writer.uint(vertex_size, offset["index"])
                writer.vec(offset["pos"])
            elif kind == 2:
                writer.sint(bone_size, offset["index"])
                writer.vec(offset["pos"])
                writer.vec(offset["rot"])
                _write_pad(writer, offset, model.bone_morph_pad)
            elif 3 <= kind <= 7:
                writer.uint(vertex_size, offset["index"])
                writer.vec(offset["uv"])
            elif kind == 8:
                writer.sint(material_size, offset["index"])
                writer.u8(offset["op"])
                writer.vec(offset["diffuse"])
                writer.vec(offset["specular"])
                writer.f32(offset["shininess"])
                writer.vec(offset["ambient"])
                writer.vec(offset["edge_color"])
                writer.f32(offset["edge_size"])
                writer.vec(offset["tex_coef"])
                writer.vec(offset["env_coef"])
                writer.vec(offset["toon_coef"])
                _write_pad(writer, offset, model.material_morph_pad)
            elif kind == 10:
                writer.sint(rigid_size, offset["index"])
                writer.u8(offset["local"])
                writer.vec(offset["velocity"])
                writer.vec(offset["torque"])
            else:
                raise PmxError("unknown morph type %d" % kind)

    writer.i32(len(model.display_frames))
    for frame in model.display_frames:
        writer.text(encoding, frame["name"])
        writer.text(encoding, frame["name_en"])
        writer.u8(frame["special"])
        writer.i32(len(frame["items"]))
        for kind, index in frame["items"]:
            writer.u8(kind)
            writer.sint(bone_size if kind == 0 else morph_size, index)

    writer.i32(len(model.rigid_bodies))
    for body in model.rigid_bodies:
        writer.text(encoding, body["name"])
        writer.text(encoding, body["name_en"])
        writer.sint(bone_size, body["bone"])
        writer.u8(body["group"])
        writer.u16(body["no_collide"])
        writer.u8(body["shape"])
        writer.vec(body["size"])
        writer.vec(body["pos"])
        writer.vec(body["rot"])
        writer.f32(body["mass"])
        writer.f32(body["lin_damp"])
        writer.f32(body["ang_damp"])
        writer.f32(body["restitution"])
        writer.f32(body["friction"])
        writer.u8(body["mode"])

    writer.i32(len(model.joints))
    for joint in model.joints:
        writer.text(encoding, joint["name"])
        writer.text(encoding, joint["name_en"])
        writer.u8(joint["type"])
        writer.sint(rigid_size, joint["a"])
        writer.sint(rigid_size, joint["b"])
        writer.vec(joint["pos"])
        writer.vec(joint["rot"])
        writer.vec(joint["pos_lo"])
        writer.vec(joint["pos_hi"])
        writer.vec(joint["rot_lo"])
        writer.vec(joint["rot_hi"])
        writer.vec(joint["spring_pos"])
        writer.vec(joint["spring_rot"])

    # Soft bodies are a 2.1 section and deliberately not written.
    return bytes(writer.buf)


def save(model, path):
    with open(path, "wb") as handle:
        handle.write(dumps(model))


# ---------------------------------------------------------------------------
# cross-check helper
# ---------------------------------------------------------------------------

def first_difference(left, right):
    """Offset of the first differing byte, or None when the buffers match."""
    limit = min(len(left), len(right))
    for offset in range(limit):
        if left[offset] != right[offset]:
            return offset
    return None if len(left) == len(right) else limit
