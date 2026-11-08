"""Sidebar panel."""

import bpy

from . import operators


class TOARUIF_PT_panel(bpy.types.Panel):
    bl_idname = "TOARUIF_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "幻想收束"
    bl_label = "模型工具"

    def draw(self, context):
        layout = self.layout
        ready = operators.mmd_operator("import_model") is not None

        box = layout.box()
        if ready:
            box.label(text="mmd_tools 已就绪", icon="CHECKMARK")
        else:
            box.label(text="未检测到 mmd_tools", icon="ERROR")
            box.label(text="请先在偏好设置中安装并启用")

        column = layout.column(align=True)
        column.scale_y = 1.4
        column.operator("toaruif.import_model", text="导入模型 (PMX)", icon="IMPORT")

        column = layout.column(align=True)
        column.scale_y = 1.4
        column.operator("toaruif.import_vmd", text="导入动作 (VMD)", icon="ANIM")
        layout.label(text="动作导入前请先选中模型或其骨架", icon="INFO")

        box = layout.box()
        box.label(text="卡通渲染", icon="MATERIAL")
        box.label(text="需先选中模型；只替换材质与描边，不动环境", icon="INFO")
        column = box.column(align=True)
        column.scale_y = 1.4
        thin = column.operator("toaruif.setup_toon", text="细描边（线宽恒定）",
                               icon="MATERIAL")
        thin.outline_bold = False
        thin.setup_scene = False
        bold = column.operator("toaruif.setup_toon", text="粗描边（游戏近景）",
                               icon="MATERIAL")
        bold.outline_bold = True
        bold.setup_scene = False
        box.label(text="细描边：像素恒定线宽，放大不变粗", icon="INFO")
        box.label(text="粗描边：像素线宽 + 模型空间厚度，近景更粗（游戏公式）")
        box.label(text="F9 可改分界偏移/遮罩强度/边缘光/光照随镜头")
        box.label(text="勾选「设置环境」才会切换引擎/世界/灯光/相机")


def register():
    bpy.utils.register_class(TOARUIF_PT_panel)


def unregister():
    bpy.utils.unregister_class(TOARUIF_PT_panel)
