"""《魔法禁书目录 幻想收束》模型多功能处理工具 —— 导入、动作与卡通渲染。

导入模型：把 PMX 2.1 模型降级为 PMX 2.0 再用 mmd_tools 导入，随后自动
「装载变形」把表情绑到滑杆，并把原本的 flip 归纳表情还原成瞬间切换。
导入动作：按当前选中的模型找到根节点后导入 VMD。
一键卡通渲染：按 Ter/Character/Toon*Ter 的真实算式复现——世界空间固定角色光
dot(N, _CharaLightDir) + _LambertOffset 做 0.07 宽的硬分界，_mask 绿通道以全强度
推动分界（0 恒暗、1 恒亮），再按 _ToonShadeMask 合成受光/环境两项，并用 albedo⁴
保留亮色环境光；边缘光、描边颜色与宽度公式同样取自 shader。描边是几何节点的
反向外壳，默认像素恒宽；粗描边再把固定模型空间厚度叠加上去（还原游戏近景变粗）。
默认只改材质与描边，引擎、视图变换、世界、灯光、相机要在 F9 勾选「设置环境」才会一并设置。

安装：把这个目录压缩为 zip，在 Blender 的「编辑 → 偏好设置 → 添加-插件 /
扩展 → 从磁盘安装」中选择该 zip。
"""

bl_info = {
    "name": "幻想收束模型工具 (ToaruIF Model Tools)",
    "author": "ToaruIF Model",
    "version": (0, 5, 6),
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
