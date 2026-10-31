"""PMX 2.1 -> PMX 2.0 downgrade for the ToaruIF character pipeline.

The game ships PMX 2.1 models whose facial expressions are *flip* morphs.
Neither mmd_tools nor MMD itself understands PMX 2.1, and a flip morph has no
2.0 equivalent that behaves the same way on its own:

* PMX 2.1 flip: ``index = (int)((count + 1) * value) - 1`` and the selected
  morph is forced to the fixed weight stored in the flip.  With one target that
  is a hard binary switch at 0.5.
* PMX 2.0 group: every entry fades in linearly with its ``factor``.

So the record is rewritten as a group morph (identical payload layout, one type
byte) and the importer later attaches a step driver to each entry, which restores
the hard switch.  This module reports exactly which morphs were converted and
what fixed weight each entry carries, so the driver can be built later.

Everything else that is 2.1-only (SoftBody section, QDEF weights, impulse morphs)
is either dropped or downgraded with a warning.
"""

import os

from . import pmx_io

__all__ = ["PmxConversionError", "ConversionResult", "convert_file",
           "downgrade_model", "is_flip_morph"]

FLIP_TYPE = 9
GROUP_TYPE = 0
IMPULSE_TYPE = 10

# Material draw flags introduced by 2.1.  2.0 readers ignore them, but the
# underlying geometry (point / line primitives, vertex colour) will not draw.
MATERIAL_21_FLAGS = (0x20, 0x40, 0x80)


class PmxConversionError(Exception):
    """A 2.1 feature that cannot be represented in 2.0 at all."""


def is_flip_morph(morph):
    return morph["type"] == FLIP_TYPE


class ConversionResult:
    """Outcome of a 2.1 -> 2.0 downgrade."""

    def __init__(self):
        self.path = ""
        self.source_version = 2.0
        self.converted = False
        self.flips = []          # [{"name", "targets": [{"index", "weight"}]}]
        self.warnings = []

    def flip_names(self):
        return [flip["name"] for flip in self.flips]


def _check_representable(model):
    """Raise for 2.1 features that have no 2.0 equivalent at all."""
    impulse = [morph["name"] for morph in model.morphs
               if morph["type"] == IMPULSE_TYPE]
    if impulse:
        raise PmxConversionError(
            "该模型含有 PMX 2.1 的冲击(impulse)表情，PMX 2.0 没有对应类型，无法安全转换：%s"
            % ", ".join(impulse[:6]))


def downgrade_model(model, warnings):
    """Downgrade a parsed model in place and return the flip descriptions."""
    _check_representable(model)
    flips = []

    for morph in model.morphs:
        if morph["type"] != FLIP_TYPE:
            continue
        flips.append({
            "name": morph["name"],
            "panel": morph["panel"],
            "targets": [{"index": offset["index"], "weight": offset["weight"]}
                        for offset in morph["offsets"]],
        })
        # Flip and group records share the same on-disk payload.
        morph["type"] = GROUP_TYPE

    qdef_count = 0
    for vertex in model.vertices:
        if vertex["weight_type"] == 4:
            vertex["weight_type"] = 2
            qdef_count += 1
    if qdef_count:
        warnings.append("顶点中有 %d 个使用 QDEF 权重，2.0 无此类型，已按 BDEF4 写出"
                        "（两者数据布局相同，变形方式不同）" % qdef_count)

    if model.soft_bodies:
        warnings.append("模型含有 %d 个 PMX 2.1 软体，PMX 2.0 无此段，已丢弃"
                        % len(model.soft_bodies))
        model.soft_bodies = []

    flagged = sum(1 for material in model.materials
                  if material["flag"] & sum(MATERIAL_21_FLAGS))
    if flagged:
        warnings.append("%d 个材质使用了 PMX 2.1 的顶点色/点/线绘制标志，"
                        "2.0 渲染会忽略这些标志" % flagged)

    model.version = 2.0
    return flips


def needs_conversion(model):
    """True when the file is not already a plain PMX 2.0 model."""
    return (model.version != 2.0
            or bool(model.soft_bodies)
            or any(morph["type"] == FLIP_TYPE for morph in model.morphs))


def default_output_path(source_path):
    stem, _ext = os.path.splitext(source_path)
    if stem.lower().endswith("_pmx20"):
        return source_path
    return stem + "_pmx20.pmx"


def convert_file(source_path, output_path=None):
    """Load ``source_path``, downgrade it to 2.0 and write the result.

    The output is written next to the source by default, which matters because
    PMX texture paths are relative to the model file and mmd_tools resolves them
    against the directory it imports from.
    """
    model = pmx_io.load(source_path)
    result = ConversionResult()
    result.source_version = model.version

    if not needs_conversion(model):
        result.path = source_path
        return result

    result.warnings = []
    result.flips = downgrade_model(model, result.warnings)
    result.converted = True

    target = output_path or default_output_path(source_path)
    data = pmx_io.dumps(model)

    if os.path.exists(target):
        with open(target, "rb") as handle:
            existing = handle.read()
        if existing == data:
            result.path = target
            return result
    else:
        existing = None

    with open(target, "wb") as handle:
        handle.write(data)
    result.path = target
    _verify(target, model)
    return result


def _verify(path, expected):
    """Re-read the written file and confirm the important counts survived."""
    check = pmx_io.load(path)
    problems = []
    if check.version != 2.0:
        problems.append("version=%r" % check.version)
    for label, a, b in (
        ("vertices", len(check.vertices), len(expected.vertices)),
        ("bones", len(check.bones), len(expected.bones)),
        ("morphs", len(check.morphs), len(expected.morphs)),
        ("materials", len(check.materials), len(expected.materials)),
    ):
        if a != b:
            problems.append("%s %d != %d" % (label, a, b))
    if problems:
        raise PmxConversionError("转换后的文件校验失败：%s" % "; ".join(problems))
