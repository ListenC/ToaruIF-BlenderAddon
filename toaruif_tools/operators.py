"""Import operators: PMX model (with the 2.1 -> 2.0 downgrade) and VMD motion."""

import os

import bpy
from bpy_extras.io_utils import ImportHelper

from . import converter, toon

# mmd_tools has lived under three module names: a legacy add-on, a Blender 4.2
# user extension and the blender.org extension.  Any of them may be installed.
MMD_MODULES = (
    "bl_ext.blender_org.mmd_tools",
    "bl_ext.user_default.mmd_tools",
    "mmd_tools",
)

MMD_MISSING = ("未找到 mmd_tools（MMD Tools）。\n"
               "请先在 Blender 的「编辑 → 偏好设置 → 扩展 / 插件」里安装并启用它，然后重试。")

MODEL_SCALE = 0.08
VMD_SCALE = 0.08


def file_selection(path):
    """Operator arguments that satisfy every mmd_tools generation.

    Older releases iterate ``files`` unconditionally (an empty collection
    silently imports nothing), newer ones accept a bare ``filepath``.  Setting
    both keeps the call working across 3.6 - 5.x.
    """
    return {
        "filepath": path,
        "directory": os.path.dirname(path),
        "files": [{"name": os.path.basename(path)}],
    }


# ---------------------------------------------------------------------------
# mmd_tools bridge
# ---------------------------------------------------------------------------

def mmd_operator(name):
    """The mmd_tools operator, or None when the add-on is not enabled.

    ``getattr`` alone is not enough: Blender hands back a placeholder operator
    object even for unregistered modules and only fails once it is called, so
    the RNA type is resolved to prove the operator really exists.
    """
    try:
        operator = getattr(bpy.ops.mmd_tools, name)
        operator.get_rna_type()
        return operator
    except Exception:  # noqa: BLE001 - any lookup failure means "not there"
        return None


def ensure_mmd_tools():
    """Enable mmd_tools if it is installed but disabled.  Returns a bool."""
    if mmd_operator("import_model") is not None:
        return True
    for module in MMD_MODULES:
        try:
            bpy.ops.preferences.addon_enable(module=module)
        except Exception:  # noqa: BLE001 - module not installed
            continue
        if mmd_operator("import_model") is not None:
            return True
    return False


def _popup(message, title, icon):
    """Best effort modal hint; the report alone is easy to miss in the status bar."""
    if bpy.app.background:
        return
    try:
        wm = bpy.context.window_manager

        def draw(self, _context):
            self.layout.label(text=message)

        wm.popup_menu(draw, title=title, icon=icon)
    except Exception:  # noqa: BLE001
        pass


def _fail(operator, message):
    operator.report({"ERROR"}, message)
    _popup(message, "幻想收束模型工具", "ERROR")
    return {"CANCELLED"}


# ---------------------------------------------------------------------------
# scene helpers
# ---------------------------------------------------------------------------

def iter_hierarchy(obj):
    yield obj
    for child in obj.children:
        yield from iter_hierarchy(child)


def find_root(obj):
    """Walk up the parent chain to the mmd_tools ROOT empty."""
    while obj is not None:
        if getattr(obj, "mmd_type", "") == "ROOT":
            return obj
        obj = obj.parent
    return None


def root_of_context(context):
    """The model root implied by the current selection."""
    obj = context.view_layer.objects.active
    if obj is None:
        selected = list(context.selected_objects)
        obj = selected[0] if selected else None
    root = find_root(obj)
    if root is not None:
        return root
    roots = [o for o in context.scene.objects if getattr(o, "mmd_type", "") == "ROOT"]
    return roots[0] if len(roots) == 1 else None


def select_only(context, obj):
    for other in context.view_layer.objects:
        try:
            other.select_set(False)
        except RuntimeError:
            pass
    try:
        obj.select_set(True)
    except RuntimeError:
        return False
    context.view_layer.objects.active = obj
    return True


def placeholder_of(root):
    for obj in iter_hierarchy(root):
        if obj.type == "MESH" and getattr(obj, "mmd_type", "") == "PLACEHOLDER":
            return obj
    return None


def _path_quote(name):
    return name.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# flip -> step driver
# ---------------------------------------------------------------------------

def step_expression(target_count, position, weight):
    """Python driver expression reproducing PMX 2.1 flip semantics.

    PMX picks entry ``j`` when ``(j+1)/(n+1) <= value < (j+2)/(n+1)``, the last
    entry also matching at value 1.0, and forces its weight to a fixed number.
    A group entry contributes ``morph_value * factor``, so setting the factor to
    ``weight / v`` inside that window and to 0 outside reproduces the switch --
    with a single target this is the hard 0.5 threshold the game uses.
    """
    lo = (position + 1.0) / (target_count + 1.0)
    hi = (position + 2.0) / (target_count + 1.0)
    last = position == target_count - 1
    if last:
        condition = "v >= %.6f" % lo
    else:
        condition = "(v >= %.6f) and (v < %.6f)" % (lo, hi)
    return "(%r / v) if %s else 0.0" % (float(weight), condition)


def apply_step_drivers(root, flips):
    """Attach the step drivers to the group morphs converted from flip morphs."""
    placeholder = placeholder_of(root)
    if placeholder is None or placeholder.data.shape_keys is None:
        return 0
    sliders = placeholder.data.shape_keys.key_blocks
    groups = {group.name: group for group in root.mmd_root.group_morphs}
    count = 0
    for flip in flips:
        group = groups.get(flip["name"])
        if group is None:
            continue
        entries = list(group.data)
        targets = flip["targets"]
        total = len(targets)
        if not total or len(entries) != total:
            # A mismatch means mmd_tools regrouped the entries; leaving the
            # linear factor in place is safer than driving the wrong morph.
            continue
        data_path = 'data.shape_keys.key_blocks["%s"].value' % _path_quote(group.name)
        for position, (entry, target) in enumerate(zip(entries, targets)):
            driver = entry.driver_add("factor").driver
            for variable in list(driver.variables):
                driver.variables.remove(variable)
            variable = driver.variables.new()
            variable.name = "v"
            variable.type = "SINGLE_PROP"
            prop = variable.targets[0]
            prop.id_type = "OBJECT"
            prop.id = placeholder
            prop.data_path = data_path
            driver.expression = step_expression(total, position, target["weight"])
            count += 1
    return count


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class TOARUIF_OT_import_model(bpy.types.Operator, ImportHelper):
    bl_idname = "toaruif.import_model"
    bl_label = "导入模型"
    bl_description = ("导入《魔法禁书目录 幻想收束》的 PMX 模型：先降级为 PMX 2.0，"
                      "再用 mmd_tools 导入并自动装载表情")
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".pmx"
    filter_glob: bpy.props.StringProperty(default="*.pmx", options={"HIDDEN"})

    bind_expressions: bpy.props.BoolProperty(
        name="装载表情",
        description="导入后绑定变形（装载）并把由 flip 转换来的表情组还原为瞬间切换",
        default=True,
    )

    def execute(self, context):
        if not ensure_mmd_tools():
            return _fail(self, MMD_MISSING)

        source = self.filepath
        if not os.path.isfile(source):
            return _fail(self, "找不到文件：%s" % source)

        try:
            result = converter.convert_file(source)
        except converter.PmxConversionError as exc:
            return _fail(self, str(exc))
        except Exception as exc:  # noqa: BLE001 - parser errors are user facing
            return _fail(self, "无法转换 PMX：%s" % exc)

        before = set(bpy.data.objects)
        try:
            bpy.ops.mmd_tools.import_model(
                **file_selection(result.path),
                types={"MESH", "ARMATURE", "MORPHS"},
                scale=MODEL_SCALE,
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(self, "mmd_tools 导入失败：%s" % exc)

        roots = [obj for obj in bpy.data.objects
                 if obj not in before and getattr(obj, "mmd_type", "") == "ROOT"]
        if not roots:
            return _fail(self, "导入完成，但找不到模型根节点；请手动检查。")
        root = roots[0]

        drivers = 0
        if self.bind_expressions:
            select_only(context, root)
            try:
                bpy.ops.mmd_tools.morph_slider_setup(type="BIND")
                drivers = apply_step_drivers(root, result.flips)
            except Exception as exc:  # noqa: BLE001
                self.report({"WARNING"}, "装载表情失败：%s" % exc)

        for warning in result.warnings:
            self.report({"WARNING"}, warning)

        message = "已导入 %s" % root.name
        if result.converted:
            message += "（已转换为 PMX 2.0，%d 个表情组还原为 flip 行为）" % drivers
        self.report({"INFO"}, message)
        return {"FINISHED"}


class TOARUIF_OT_import_vmd(bpy.types.Operator, ImportHelper):
    bl_idname = "toaruif.import_vmd"
    bl_label = "导入动作"
    bl_description = ("按当前选中的对象找到模型根节点，再交由 mmd_tools 导入 VMD 动作")
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".vmd"
    filter_glob: bpy.props.StringProperty(default="*.vmd", options={"HIDDEN"})

    def execute(self, context):
        if not ensure_mmd_tools():
            return _fail(self, MMD_MISSING)

        root = root_of_context(context)
        if root is None:
            return _fail(self, "没有选中任何幻想收束模型。请先选中模型（或它的骨架/网格），再导入动作。")

        if not os.path.isfile(self.filepath):
            return _fail(self, "找不到文件：%s" % self.filepath)

        select_only(context, root)
        before = set(bpy.data.actions)
        try:
            bpy.ops.mmd_tools.import_vmd(**file_selection(self.filepath),
                                         scale=VMD_SCALE)
        except Exception as exc:  # noqa: BLE001
            return _fail(self, "mmd_tools 导入动作失败：%s" % exc)

        actions = [action for action in bpy.data.actions if action not in before]
        used = [action for action in actions if action.frame_range[1] > action.frame_range[0]]
        if used:
            start = min(int(action.frame_range[0]) for action in used)
            end = max(int(action.frame_range[1]) for action in used)
            scene = context.scene
            scene.frame_start = start
            scene.frame_end = end
            scene.frame_set(start)
            track_count = max(len(action.fcurves) for action in used)
            message = "已导入动作 %s：%d 条曲线，帧范围 %d - %d" % (
                os.path.basename(self.filepath), track_count, start, end)
        else:
            message = ("动作文件已读取，但没有产生任何动画曲线。"
                       "请确认该 VMD 含有与当前模型匹配的骨骼或表情名称。")
            self.report({"WARNING"}, message)
            return {"CANCELLED"}

        self.report({"INFO"}, message)
        return {"FINISHED"}


class TOARUIF_OT_setup_toon(bpy.types.Operator):
    bl_idname = "toaruif.setup_toon"
    bl_label = "一键卡通渲染"
    bl_description = ("把当前模型换成游戏风格的卡通着色：相机空间竖直条带做硬明暗分界、"
                      "_mask 通道推动分界，再按游戏颜色套 mmd_tools 描边")
    bl_options = {"REGISTER", "UNDO"}

    outline: bpy.props.BoolProperty(
        name="添加描边",
        description="用 mmd_tools 的边缘预览加反向外壳描边（游戏同款暖褐色，约 1.8mm）",
        default=True,
    )
    outline_width: bpy.props.FloatProperty(
        name="描边宽度",
        description="描边厚度（米）。1.8mm 约等于正常构图下的 1-2 像素",
        default=toon.OUTLINE_WIDTH,
        min=0.0,
        max=0.02,
        precision=4,
        subtype="DISTANCE",
    )
    outline_color: bpy.props.FloatVectorProperty(
        name="描边颜色",
        description="描边颜色，默认是游戏 _OutlineColor 的暖褐色",
        default=toon.OUTLINE_COLOR,
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
    )
    light_direction: bpy.props.FloatProperty(
        name="受光方向",
        description="明暗分界绕竖轴旋转：0.5 为正面受光，0.42 与 BV-Toon 预设一致",
        default=toon.LIGHT_DIRECTION,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    break_width: bpy.props.FloatProperty(
        name="分界宽度",
        description="明暗分界的柔和度；越小越像游戏里的硬两段，越大越柔和",
        default=toon.BREAK_WIDTH,
        min=0.001,
        max=0.5,
        subtype="FACTOR",
    )
    break_centre: bpy.props.FloatProperty(
        name="分界位置",
        description="明暗分界落在哪里：越小阴影面积越大，越大越靠近剪影；"
                    "0.1 是明显的两段卡通效果",
        default=toon.BREAK_CENTRE,
        min=-0.5,
        max=1.0,
        subtype="FACTOR",
    )
    mask_strength: bpy.props.FloatProperty(
        name="遮罩强度",
        description="对应 _ToonShadeMask：_mask 绿通道对明暗分界的推动力度"
                    "（0 = 只看光照，默认 0.35 与游戏材质一致）",
        default=toon.TOON_SHADE_MASK,
        min=0.0,
        max=1.0,
    )
    rim_strength: bpy.props.FloatProperty(
        name="边缘光强度",
        description="对应 _RimScale：游戏材质里是 0，由 SetRimSettings 在运行时上调；"
                    "保持 0 即与出厂状态一致",
        default=toon.RIM_SCALE,
        min=0.0,
        max=1.0,
    )

    def execute(self, context):
        root = root_of_context(context)
        if root is None:
            return _fail(self, "没有选中任何幻想收束模型。请先选中模型（或它的骨架/网格）。")

        meshes = [obj for obj in iter_hierarchy(root)
                  if obj.type == "MESH" and getattr(obj, "mmd_type", "") != "PLACEHOLDER"]
        if not meshes:
            return _fail(self, "这个模型里没有找到可着色的网格。")

        shaded = 0
        plain = 0
        masks_found = 0
        fallback = []
        for obj in meshes:
            for slot in obj.material_slots:
                material = slot.material
                if material is None:
                    continue
                if material.name.startswith(toon.EDGE_MATERIAL_PREFIX):
                    continue
                base = toon.base_image_of(material)
                if base is None:
                    plain += 1
                    continue
                part = toon.part_of(base.name)
                style = toon.STYLES[part]
                mask = toon.mask_image_for(base, part)
                if mask is None:
                    mask = toon.ensure_default_mask()
                    fallback.append(base.name)
                else:
                    masks_found += 1
                toon.build_toon_material(material, base, mask, part, style,
                                         self.mask_strength,
                                         light_direction=self.light_direction,
                                         break_width=self.break_width,
                                         break_centre=self.break_centre,
                                         rim_scale=self.rim_strength)
                shaded += 1

        if shaded == 0:
            return _fail(self, "所有材质都没有基础颜色贴图，无法套用卡通渲染。")

        outline_mode = None
        if self.outline:
            outline_mode = toon.setup_outline(context, meshes, self.outline_width,
                                              tuple(self.outline_color))
        else:
            toon.clear_outline(context, meshes)

        engine = toon.ensure_engine(context.scene)
        toon.ensure_view_transform(context.scene)
        toon.ensure_world(context.scene)
        toon.ensure_light(context.scene)
        toon.ensure_camera(context.scene, meshes)

        message = "已应用卡通渲染：%d 个材质（%s）" % (shaded, engine)
        if outline_mode == "mmd":
            message += "，描边由 mmd_tools 边缘预览生成"
        elif outline_mode == "hull":
            message += "，mmd_tools 描边不可用，改用自建外壳"
        if fallback:
            self.report({"WARNING"},
                        "%d 个材质没有找到 _mask 贴图，改用中性遮罩：%s"
                        % (len(fallback), "、".join(sorted(set(fallback))[:4])))
        if plain:
            self.report({"WARNING"}, "%d 个没有贴图的材质已跳过" % plain)
        self.report({"INFO"}, message + "，命中 _mask %d 个" % masks_found)
        return {"FINISHED"}


CLASSES = (TOARUIF_OT_import_model, TOARUIF_OT_import_vmd, TOARUIF_OT_setup_toon)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
