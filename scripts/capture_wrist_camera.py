"""从工作台腕部或场景相机采集一帧原始分辨率 RGB，并保存为 PNG。"""

import argparse
from pathlib import Path
import struct
import zlib

import mujoco
import numpy as np


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """用标准库保存 uint8、HWC 排列的 RGB 图像。"""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB 图像必须为 uint8，形状为 (height, width, 3)")
    if path.suffix.lower() != ".png":
        raise ValueError(f"输出文件必须使用 .png 后缀：{path}")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", checksum)

    height, width, _ = rgb.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in rgb)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def capture_camera(scene_path: Path, output_path: Path, camera_name: str) -> None:
    """加载 home 姿态，并按相机声明的分辨率采集 RGB。"""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ur5e/home").id)
    mujoco.mj_forward(model, data)
    camera_id = model.camera(camera_name).id
    width, height = (int(value) for value in model.cam_resolution[camera_id])
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera_id)
        rgb = renderer.render()
    save_rgb_png(output_path, rgb)
    print(f"{camera_name} RGB 已保存：{output_path}，分辨率 {width}×{height}，数组形状 {rgb.shape}")


def capture_wrist_camera(scene_path: Path, output_path: Path) -> None:
    """保留腕部采集入口，默认使用腕部相机。"""
    capture_camera(scene_path, output_path, "ur5e/wrist_camera")


def main() -> None:
    """解析输出路径并采集腕部图像。"""
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("wrist", "scene"), default="wrist")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    camera_name = "ur5e/wrist_camera" if args.camera == "wrist" else "scene_camera"
    output_path = args.output or project_dir / "outputs" / f"{args.camera}_camera.png"
    capture_camera(project_dir / "assets" / "scenes" / "workbench.xml", output_path, camera_name)


if __name__ == "__main__":
    main()
