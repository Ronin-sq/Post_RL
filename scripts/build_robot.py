"""将 Menagerie 的 UR5e 与 Robotiq 2F-85 装配为可直接加载的 MJCF。"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# 本工作台采用的关节角度范围；控制器与物理关节使用相同限位。
ARM_JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan_joint": (-360.0, 360.0),
    "shoulder_lift_joint": (-360.0, 360.0),
    "elbow_joint": (-160.0, 160.0),
    "wrist_1_joint": (-360.0, 360.0),
    "wrist_2_joint": (-360.0, 360.0),
    "wrist_3_joint": (-360.0, 360.0),
}

WRIST_CAMERA_RESOLUTION = (640, 480)


def build_robot(project_dir: Path) -> Path:
    """在腕部安装夹爪，并将带相对网格路径的组合模型保存到 assets/robots。"""
    menagerie_dir = project_dir / "third_party" / "mujoco_menagerie"
    arm_path = menagerie_dir / "universal_robots_ur5e" / "ur5e.xml"
    gripper_path = menagerie_dir / "robotiq_2f85" / "2f85.xml"
    output_path = project_dir / "assets" / "robots" / "ur5e_2f85.xml"

    arm = mujoco.MjSpec.from_file(str(arm_path))
    gripper = mujoco.MjSpec.from_file(str(gripper_path))
    attachment_site = arm.site("attachment_site")
    if attachment_site is None:
        raise ValueError(f"机械臂缺少 attachment_site：{arm_path}")

    # 导出后的模型需要同时引用两个上游目录中的网格。
    mesh_paths = {
        mesh.name: arm_path.parent / "assets" / mesh.file
        for mesh in arm.meshes
    }
    mesh_paths.update({
        f"gripper/{mesh.name}": gripper_path.parent / "assets" / mesh.file
        for mesh in gripper.meshes
    })

    arm.modelname = "ur5e_2f85"
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10
    for joint_name, limits_deg in ARM_JOINT_LIMITS_DEG.items():
        limits_rad = np.deg2rad(limits_deg)
        joint = arm.joint(joint_name)
        joint.limited = True
        joint.range = limits_rad
        joint.margin = np.deg2rad(0.5)
        joint.solref_limit = [0.004, 1.0]
        joint.solimp_limit = [0.99, 0.999, 0.001, 0.5, 2.0]
        actuator = arm.actuator(joint_name.removesuffix("_joint"))
        actuator.ctrllimited = True
        actuator.ctrlrange = limits_rad

    # 保留夹爪原有安装偏移；MjSpec 同时装配约束、肌腱和控制器。
    arm.attach(gripper, prefix="gripper/", site=attachment_site)
    # 相机位于腕部侧面，避开夹爪主体；光轴朝向夹爪前方的抓取区域。
    # MuJoCo 相机沿自身 -Z 看，xyaxes 分别定义图像的右方与上方。
    arm.body("wrist_3_link").add_camera(
        name="wrist_camera",
        pos=[0.0, 0.055, 0.075],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.075, 0.345],
        fovy=60.0,
        resolution=WRIST_CAMERA_RESOLUTION,
    )
    arm.compile()

    root = ET.fromstring(arm.to_xml())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    # 让场景按实际关键帧计算容量，避免子模型导出的 nkey 与父模型冲突。
    size = root.find("size")
    if size is not None:
        size.attrib.pop("nkey", None)
        if not size.attrib:
            root.remove(size)
    for mesh in root.findall("asset/mesh"):
        source_path = mesh_paths[mesh.attrib["name"]]
        relative_path = Path(os.path.relpath(source_path, output_path.parent))
        mesh.set("file", relative_path.as_posix())

    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "<!-- 由 scripts/build_robot.py 生成；修改上游模型后请重新运行装配脚本。 -->\n"
    output_path.write_text(header + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    """生成工作台使用的机械臂与夹爪组合模型。"""
    project_dir = Path(__file__).resolve().parents[1]
    output_path = build_robot(project_dir)
    print(f"组合模型已生成：{output_path}")


if __name__ == "__main__":
    main()
