"""《魔法禁书目录 幻想收束》模型多功能处理工具 —— 导入、动作与卡通渲染。

导入模型：把 PMX 2.1 模型降级为 PMX 2.0 再用 mmd_tools 导入，随后自动
「装载变形」把表情绑到滑杆，并把原本的 flip 归纳表情还原成瞬间切换。
导入动作：按当前选中的模型找到根节点后导入 VMD。
一键卡通渲染：参考 BV-Toon 的相机空间竖直条带做硬明暗分界，保留游戏 shader
的 _ToonShadeMask（_mask 绿通道推分界）与 _RimScale；描边是几何节点的反向
外壳，线宽按像素恒定，镜头拉近也不会变粗，颜色取 _OutlineColor。

安装：把这个目录压缩为 zip，在 Blender 的「编辑 → 偏好设置 → 添加-插件 /
扩展 → 从磁盘安装」中选择该 zip。
"""

bl_info = {
    "name": "幻想收束模型工具 (ToaruIF Model Tools)",
    "author": "ToaruIF Model",
    "version": (0, 5, 1),
    "blender": (3, 6, 0),
    "location": "3D 视图 > 侧栏 > 幻想收束",
    "description": "《魔法禁书目录 幻想收束》模型工具：PMX 2.1 降级导入、表情装载、VMD 动作、一键卡通渲染",
    "category": "Import-Export",
}

import bpy

from . import operators, ui

modules = (operators, ui)


def register():
    for module in modules:
        module.register()


def unregister():
    for module in reversed(modules):
        module.unregister()
