"""键盘遥操 UR5e：WSAD 平移，I/J 升降，Q/E 绕世界 Z 轴旋转，空格开合夹爪。"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

if __package__:
    from .teleop_window import TeleopWindow
else:
    from teleop_window import TeleopWindow


ARM_TARGET_SPEED_RAD_PER_SECOND = np.deg2rad(60.0)
GRIPPER_TARGET_SPEED_PER_SECOND = 255.0


class KeyboardTeleop:
    """保存末端位姿目标，通过阻尼逆运动学将平移和旋转转换为关节目标。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        control_targets: np.ndarray,
        *,
        step_meters: float = 0.005,
        fine_step_meters: float = 0.001,
        speed_meters_per_second: float = 0.05,
        fine_speed_meters_per_second: float = 0.01,
        yaw_speed_rad_per_second: float = np.deg2rad(30.0),
        fine_yaw_speed_rad_per_second: float = np.deg2rad(5.0),
    ) -> None:
        if not np.isfinite(step_meters) or step_meters <= 0:
            raise ValueError("移动步长必须是正的有限数值")
        if not np.isfinite(fine_step_meters) or fine_step_meters <= 0:
            raise ValueError("精调步长必须是正的有限数值")
        if not np.isfinite(speed_meters_per_second) or speed_meters_per_second <= 0:
            raise ValueError("移动速度必须是正的有限数值")
        if not np.isfinite(fine_speed_meters_per_second) or fine_speed_meters_per_second <= 0:
            raise ValueError("精调速度必须是正的有限数值")
        if not np.isfinite(yaw_speed_rad_per_second) or yaw_speed_rad_per_second <= 0:
            raise ValueError("旋转速度必须是正的有限数值")
        if not np.isfinite(fine_yaw_speed_rad_per_second) or fine_yaw_speed_rad_per_second <= 0:
            raise ValueError("精调旋转速度必须是正的有限数值")
        self.model = model
        self.ik_data = mujoco.MjData(model)
        self.site_id = model.site("ur5e/gripper/pinch").id
        self.gripper_id = model.actuator("ur5e/gripper/fingers_actuator").id
        self.arm_actuator_ids = np.flatnonzero(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        joint_ids = model.actuator_trnid[self.arm_actuator_ids, 0]
        self.qpos_ids = model.jnt_qposadr[joint_ids]
        self.dof_ids = model.jnt_dofadr[joint_ids]
        self.joint_ranges = model.jnt_range[joint_ids]
        self.step_meters = step_meters
        self.fine_step_meters = fine_step_meters
        self.speed_meters_per_second = speed_meters_per_second
        self.fine_speed_meters_per_second = fine_speed_meters_per_second
        self.yaw_speed_rad_per_second = yaw_speed_rad_per_second
        self.fine_yaw_speed_rad_per_second = fine_yaw_speed_rad_per_second
        self.fine_mode = False
        self.was_moving = False
        self.reset_reference(data, control_targets)

    def reset_reference(self, data: mujoco.MjData, control_targets: np.ndarray) -> None:
        """以当前控制参考的 FK 为目标起点，避免把重力下垂误差累加到新指令。"""
        mujoco.mj_copyData(self.ik_data, self.model, data)
        self.ik_data.qpos[self.qpos_ids] = control_targets[self.arm_actuator_ids]
        mujoco.mj_forward(self.model, self.ik_data)
        self.target_position = self.ik_data.site_xpos[self.site_id].copy()
        self.target_rotation = self.ik_data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.was_moving = False

    def _solve_ik(
        self,
        data: mujoco.MjData,
        control_targets: np.ndarray,
        position: np.ndarray,
        target_rotation: np.ndarray,
    ) -> bool:
        mujoco.mj_copyData(self.ik_data, self.model, data)
        self.ik_data.qpos[self.qpos_ids] = control_targets[self.arm_actuator_ids]
        jac_position = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros((3, self.model.nv))
        rotation_quat = np.zeros(4)
        rotation_error = np.zeros(3)
        for _ in range(100):
            mujoco.mj_forward(self.model, self.ik_data)
            position_error = position - self.ik_data.site_xpos[self.site_id]
            rotation = self.ik_data.site_xmat[self.site_id].reshape(3, 3)
            mujoco.mju_mat2Quat(rotation_quat, (target_rotation @ rotation.T).ravel())
            if rotation_quat[0] < 0:
                rotation_quat *= -1
            mujoco.mju_quat2Vel(rotation_error, rotation_quat, 1.0)
            if np.linalg.norm(position_error) < 5e-5 and np.linalg.norm(rotation_error) < 5e-4:
                return True

            mujoco.mj_jacSite(self.model, self.ik_data, jac_position, jac_rotation, self.site_id)
            # 旋转误差以 0.2 m/rad 加权，使位置和姿态共同参与阻尼最小二乘。
            jacobian = np.vstack((jac_position[:, self.dof_ids], 0.2 * jac_rotation[:, self.dof_ids]))
            error = np.concatenate((position_error, 0.2 * rotation_error))
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.005**2 * np.eye(6), error)
            scale = min(1.0, 0.1 / max(np.max(np.abs(delta)), 1e-12))
            self.ik_data.qpos[self.qpos_ids] = np.clip(
                self.ik_data.qpos[self.qpos_ids] + scale * delta,
                self.joint_ranges[:, 0], self.joint_ranges[:, 1],
            )
        return False

    def _intersects_fixture(self) -> bool:
        """拒绝明显穿入桌面、地面或插槽的 IK 终点；实际运动仍由碰撞求解器约束。"""
        for contact in self.ik_data.contact:
            if contact.dist >= -0.001:
                continue
            bodies = [
                self.model.body(int(self.model.geom_bodyid[index])).name
                for index in (contact.geom1, contact.geom2)
            ]
            if any(name.startswith("ur5e/") for name in bodies) and any(
                name in ("table", "socket", "world") for name in bodies
            ):
                return True
        return False

    @staticmethod
    def movement_directions(camera_rotation: np.ndarray) -> dict[int, np.ndarray]:
        """将相机左右、前后投影到桌面，升降始终沿世界 Z 轴。"""
        right = camera_rotation[:, 0].copy()
        right[2] = 0
        norm = np.linalg.norm(right)
        if norm < 1e-8:
            raise ValueError("当前相机横轴竖直，请切换到场景相机后再移动")
        right /= norm
        forward = np.cross([0.0, 0.0, 1.0], right)
        if np.dot(forward, camera_rotation[:, 1]) < 0:
            forward *= -1
        return {
            ord("W"): forward, ord("S"): -forward,
            ord("A"): -right, ord("D"): right,
            ord("I"): np.array([0.0, 0.0, 1.0]),
            ord("J"): np.array([0.0, 0.0, -1.0]),
        }

    def move_delta(
        self,
        data: mujoco.MjData,
        control_targets: np.ndarray,
        delta: np.ndarray,
        yaw_delta_radians: float = 0.0,
    ) -> str | None:
        """更新末端位姿；绕世界 Z 左乘旋转，失败时保留原位置和朝向。"""
        position = self.target_position + delta
        cosine, sine = np.cos(yaw_delta_radians), np.sin(yaw_delta_radians)
        world_yaw = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        rotation = world_yaw @ self.target_rotation
        if not self._solve_ik(data, control_targets, position, rotation):
            return "目标不可达或触及关节限位，本次移动已取消"
        if self._intersects_fixture():
            return "目标会穿入桌面或工装，本次移动已取消"
        control_targets[self.arm_actuator_ids] = self.ik_data.qpos[self.qpos_ids]
        self.target_position = position
        self.target_rotation = rotation
        return None

    def stop_motion(self, data: mujoco.MjData, control_targets: np.ndarray) -> None:
        """取消尚未执行的位姿参考，保留夹爪目标，让机械臂平稳停止。"""
        if self.was_moving:
            control_targets[self.arm_actuator_ids] = data.ctrl[self.arm_actuator_ids]
            self.reset_reference(data, control_targets)

    def move_held_keys(
        self,
        pressed: set[int],
        data: mujoco.MjData,
        control_targets: np.ndarray,
        camera_rotation: np.ndarray,
        dt: float,
    ) -> str | None:
        """按住期间每个控制周期推进目标，速度与键盘重复率无关。"""
        directions = self.movement_directions(camera_rotation)
        direction = sum((vector for key, vector in directions.items() if key in pressed), np.zeros(3))
        norm = np.linalg.norm(direction)
        yaw_direction = int(ord("Q") in pressed) - int(ord("E") in pressed)
        if norm < 1e-8 and yaw_direction == 0:
            self.stop_motion(data, control_targets)
            return None
        self.was_moving = True
        speed = self.fine_speed_meters_per_second if self.fine_mode else self.speed_meters_per_second
        yaw_speed = self.fine_yaw_speed_rad_per_second if self.fine_mode else self.yaw_speed_rad_per_second
        delta = direction / norm * speed * dt if norm >= 1e-8 else np.zeros(3)
        return self.move_delta(data, control_targets, delta, yaw_direction * yaw_speed * dt)

    def handle_key(
        self,
        keycode: int,
        data: mujoco.MjData,
        control_targets: np.ndarray,
        camera_rotation: np.ndarray,
    ) -> str | None:
        """处理开合与模式切换；保留单步接口供自动化检查与脚本调用。"""
        if keycode == ord(" "):
            control_targets[self.gripper_id] = 0.0 if control_targets[self.gripper_id] > 127.5 else 255.0
            return "夹爪张开" if control_targets[self.gripper_id] == 0 else "夹爪闭合"
        if keycode == ord("F"):
            self.fine_mode = not self.fine_mode
            speed = self.fine_speed_meters_per_second if self.fine_mode else self.speed_meters_per_second
            yaw_speed = self.fine_yaw_speed_rad_per_second if self.fine_mode else self.yaw_speed_rad_per_second
            return f"{'精调' if self.fine_mode else '常规'}模式：{speed * 1000:g} mm/s，旋转 {np.rad2deg(yaw_speed):g} °/s"
        if keycode in (ord("Q"), ord("E")):
            angle = np.deg2rad(1.0 if self.fine_mode else 5.0)
            return self.move_delta(data, control_targets, np.zeros(3), angle if keycode == ord("Q") else -angle)
        directions = self.movement_directions(camera_rotation)
        if keycode not in directions:
            return None
        step = self.fine_step_meters if self.fine_mode else self.step_meters
        return self.move_delta(data, control_targets, step * directions[keycode])


def sync_viewer_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: mujoco.viewer.Handle,
    control_targets: np.ndarray,
) -> None:
    """将 GUI 的关节位置编辑转为控制目标，保留仿真中的实际关节状态。"""
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    time_before = data.time
    applied_controls = data.ctrl.copy()
    # GUI 显示用户要求的目标；实际施加的控制参考由 step_controls 平滑推进。
    data.ctrl[:] = control_targets

    # 第二次同步将恢复后的真实姿态显示回 GUI，避免显示瞬移后的关节位置。
    for _ in range(2):
        viewer.sync()
        changed = np.abs(data.qpos - qpos_before) > 1e-10
        with viewer.lock():
            if np.any(changed):
                for actuator_id in range(model.nu):
                    if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                        continue
                    joint_id = model.actuator_trnid[actuator_id, 0]
                    qpos_id = model.jnt_qposadr[joint_id]
                    if changed[qpos_id]:
                        data.ctrl[actuator_id] = data.qpos[qpos_id]

                # 驱动关节角度也可转换成夹爪命令，其余被动连杆由约束决定。
                driver_ids = [
                    model.joint(f"ur5e/gripper/{side}_driver_joint").qposadr[0]
                    for side in ("left", "right")
                ]
                edited_driver_ids = [index for index in driver_ids if changed[index]]
                if edited_driver_ids:
                    gripper_id = model.actuator("ur5e/gripper/fingers_actuator").id
                    data.ctrl[gripper_id] = np.mean(data.qpos[edited_driver_ids]) * 255.0 / 0.8

                data.qpos[:] = qpos_before
                data.qvel[:] = qvel_before
                data.time = time_before
                mujoco.mj_forward(model, data)

            for actuator_id in range(model.nu):
                if model.actuator_ctrllimited[actuator_id]:
                    data.ctrl[actuator_id] = np.clip(
                        data.ctrl[actuator_id], *model.actuator_ctrlrange[actuator_id]
                    )
        if not np.any(changed):
            break
    control_targets[:] = data.ctrl
    data.ctrl[:] = applied_controls


def step_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_targets: np.ndarray,
    num_steps: int = 1,
) -> None:
    """限制控制参考的变化率，并通过动力学推进实际关节位置。"""
    targets = np.clip(control_targets, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    max_change = np.where(
        model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT,
        ARM_TARGET_SPEED_RAD_PER_SECOND,
        GRIPPER_TARGET_SPEED_PER_SECOND,
    ) * model.opt.timestep
    for _ in range(num_steps):
        data.ctrl[:] += np.clip(targets - data.ctrl, -max_change, max_change)
        mujoco.mj_step(model, data)


def main() -> None:
    """初始化 home 姿态，通过物理步进执行查看器中的控制指令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("overview", "wrist", "scene"), default="overview")
    parser.add_argument("--speed-mm-s", type=float, default=50.0, help="常规模式移动速度，单位 mm/s")
    parser.add_argument("--fine-speed-mm-s", type=float, default=10.0, help="精调模式移动速度，单位 mm/s")
    parser.add_argument("--yaw-speed-deg-s", type=float, default=30.0, help="绕世界 Z 轴旋转的速度，单位 °/s")
    parser.add_argument("--fine-yaw-speed-deg-s", type=float, default=5.0, help="精调模式旋转速度，单位 °/s")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    scene_path = project_dir / "assets" / "scenes" / "workbench.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ur5e/home")
    if home_id < 0:
        raise ValueError(f"工作台场景缺少 ur5e/home 关键帧：{scene_path}")

    # 同时加载 qpos 与 ctrl，避免位置控制器启动时将机械臂拉向零位。
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    control_targets = data.ctrl.copy()
    teleop = KeyboardTeleop(
        model, data, control_targets,
        speed_meters_per_second=args.speed_mm_s / 1000.0,
        fine_speed_meters_per_second=args.fine_speed_mm_s / 1000.0,
        yaw_speed_rad_per_second=np.deg2rad(args.yaw_speed_deg_s),
        fine_yaw_speed_rad_per_second=np.deg2rad(args.fine_yaw_speed_deg_s),
    )
    paused = False

    camera_name = {
        "overview": "overview",
        "wrist": "ur5e/wrist_camera",
        "scene": "scene_camera",
    }[args.camera]
    frame_steps = max(1, round(1.0 / (60.0 * model.opt.timestep)))
    frame_seconds = frame_steps * model.opt.timestep
    print("按住 W/S 前后、A/D 左右、I/J 上下、Q/E 绕世界 Z 轴正/反转，松开停止。")
    print(f"空格切换夹爪，F 切换精调（{args.speed_mm_s:g}/{args.fine_speed_mm_s:g} mm/s），P 暂停，R 复位。")
    print(f"旋转速度 {args.yaw_speed_deg_s:g} °/s，精调 {args.fine_yaw_speed_deg_s:g} °/s；俯视时 Q 逆时针、E 顺时针。")
    print("1 场景相机，2 腕部相机，3 全景，Esc 退出。")
    last_message = None
    with TeleopWindow(model, camera_name) as window:
        while window.is_running():
            frame_start = time.monotonic()
            window.poll_events()
            if not window.is_running():
                break
            camera_rotation = data.cam_xmat[window.cam.fixedcamid].reshape(3, 3)
            for keycode in window.keys.pop_commands():
                if keycode == ord("P"):
                    paused = not paused
                    window.keys.clear()
                    teleop.stop_motion(data, control_targets)
                    print("已暂停" if paused else "继续运行")
                elif keycode == ord("R"):
                    mujoco.mj_resetDataKeyframe(model, data, home_id)
                    mujoco.mj_forward(model, data)
                    control_targets[:] = data.ctrl
                    teleop.reset_reference(data, control_targets)
                    window.keys.clear()
                elif keycode in (ord("1"), ord("2"), ord("3")):
                    camera_name = {ord("1"): "scene_camera", ord("2"): "ur5e/wrist_camera", ord("3"): "overview"}[keycode]
                    window.cam.fixedcamid = model.camera(camera_name).id
                else:
                    message = teleop.handle_key(keycode, data, control_targets, camera_rotation)
                    if message:
                        print(message)
            if not paused:
                camera_rotation = data.cam_xmat[window.cam.fixedcamid].reshape(3, 3)
                message = teleop.move_held_keys(window.keys.pressed, data, control_targets, camera_rotation, frame_seconds)
                if message and message != last_message:
                    print(message)
                last_message = message
                step_controls(model, data, control_targets, num_steps=frame_steps)
                mujoco.mj_forward(model, data)
            speed = teleop.fine_speed_meters_per_second if teleop.fine_mode else teleop.speed_meters_per_second
            yaw_speed = teleop.fine_yaw_speed_rad_per_second if teleop.fine_mode else teleop.yaw_speed_rad_per_second
            # MuJoCo 内置字体使用 ASCII；详细中文原因输出到终端。
            blocked = "BLOCKED (see terminal)" if last_message else ""
            status = f"{'PAUSED' if paused else 'RUNNING'} | {speed * 1000:g} mm/s | yaw {np.rad2deg(yaw_speed):g} deg/s\n{camera_name}\n{blocked}"
            window.render(data, status)
            remaining_seconds = frame_seconds - (time.monotonic() - frame_start)
            if remaining_seconds > 0:
                time.sleep(remaining_seconds)


if __name__ == "__main__":
    main()
