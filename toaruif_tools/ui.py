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
        box.label(text="需先选中模型；会替换材质节点并按游戏色加描边", icon="INFO")
        column = box.column(align=True)
        column.scale_y = 1.4
        thin = column.operator("toaruif.setup_toon", text="细描边（线宽恒定）",
                               icon="MATERIAL")
        thin.outline_bold = False
        bold = column.operator("toaruif.setup_toon", text="粗描边（游戏近景）",
                               icon="MATERIAL")
        bold.outline_bold = True
        box.label(text="细描边：线宽以像素为单位固定，放大不变粗；", icon="INFO")
        box.label(text="粗描边：按模型空间固定粗细，近景变粗，与游戏一致")


def register():
    bpy.utils.register_class(TOARUIF_PT_panel)


def unregister():
    bpy.utils.unregister_class(TOARUIF_PT_panel)
