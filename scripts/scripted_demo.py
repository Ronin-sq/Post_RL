"""执行固定场景的抓取插孔示例，保存可回放轨迹，可选同步采集双相机 RGB。"""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import mujoco
import numpy as np

if __package__:
    from .capture_wrist_camera import save_rgb_png
    from .teleop_window import TeleopWindow
    from .view_workbench import KeyboardTeleop, step_controls
else:
    from capture_wrist_camera import save_rgb_png
    from teleop_window import TeleopWindow
    from view_workbench import KeyboardTeleop, step_controls


CONTROL_FREQUENCY_HZ = 20
TASK_INSTRUCTION = "Pick up the red block and insert it into the blue square socket."


@dataclass
class DemoResult:
    success: bool
    duration_seconds: float
    steps: int
    cube_error_meters: np.ndarray
    episode_dir: Path | None = None


class EpisodeRecorder:
    """按 obs_t、action_t、obs_{t+1} 顺序保存轨迹及可选双视角图像。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, directory: Path, renderer: mujoco.Renderer | None) -> None:
        self.model = model
        self.directory = directory
        self.renderer = renderer
        self.gripper_id = model.actuator("ur5e/gripper/fingers_actuator").id
        self.arm_actuator_ids = np.flatnonzero(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        self.arm_qpos_ids = model.jnt_qposadr[model.actuator_trnid[self.arm_actuator_ids, 0]]
        self.driver_ids = [model.joint(f"ur5e/gripper/{side}_driver_joint").qposadr[0] for side in ("left", "right")]
        self.cube_address = model.joint("cube_free_joint").qposadr[0]
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.initial_state = np.zeros(mujoco.mj_stateSize(model, self.state_spec))
        mujoco.mj_getState(model, data, self.initial_state, self.state_spec)
        self.values: dict[str, list[np.ndarray | float | str]] = {
            key: [] for key in (
                "timestamp", "state", "action", "control_targets", "applied_ctrl", "qpos", "qvel",
                "cube_pose", "tcp_position", "stage", "next_state", "next_qpos",
            )
        }

    def state(self, data: mujoco.MjData) -> np.ndarray:
        """机器人状态为六关节角与驱动角推算的夹爪闭合比例，0 开、1 闭。"""
        closure = np.clip(np.mean(data.qpos[self.driver_ids]) / 0.8, 0.0, 1.0)
        return np.concatenate((data.qpos[self.arm_qpos_ids], [closure])).astype(np.float32)

    def before_step(self, data: mujoco.MjData, controls: np.ndarray, stage: str) -> None:
        frame = len(self.values["timestamp"])
        self.values["timestamp"].append(float(data.time))
        self.values["state"].append(self.state(data))
        action = np.concatenate((controls[self.arm_actuator_ids], [controls[self.gripper_id] / 255.0]))
        self.values["action"].append(action.astype(np.float32))
        self.values["control_targets"].append(controls.copy())
        self.values["applied_ctrl"].append(data.ctrl.copy())
        self.values["qpos"].append(data.qpos.copy())
        self.values["qvel"].append(data.qvel.copy())
        self.values["cube_pose"].append(data.qpos[self.cube_address:self.cube_address + 7].copy())
        self.values["tcp_position"].append(data.site("ur5e/gripper/pinch").xpos.copy())
        self.values["stage"].append(stage)
        if self.renderer is not None:
            for label, camera in (("scene", "scene_camera"), ("wrist", "ur5e/wrist_camera")):
                self.renderer.update_scene(data, camera=camera)
                save_rgb_png(self.directory / f"rgb_{label}" / f"{frame:06d}.png", self.renderer.render())

    def after_step(self, data: mujoco.MjData) -> None:
        self.values["next_state"].append(self.state(data))
        self.values["next_qpos"].append(data.qpos.copy())

    def save(self, result: DemoResult, data: mujoco.MjData) -> None:
        arrays = {key: np.asarray(values) for key, values in self.values.items()}
        np.savez_compressed(
            self.directory / "trajectory.npz", **arrays,
            initial_state=self.initial_state, state_spec=int(self.state_spec),
            final_qpos=data.qpos.copy(), final_qvel=data.qvel.copy(), episode_success=result.success,
        )
        metadata = {
            "format_version": 1,
            "task": TASK_INSTRUCTION,
            "source": "scripted_fixed_scene",
            "mujoco_version": mujoco.__version__,
            "success": result.success,
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "physics_timestep": self.model.opt.timestep,
            "steps": result.steps,
            "duration_seconds": result.duration_seconds,
            "cube_error_meters": result.cube_error_meters.tolist(),
            "arm_joint_names": [self.model.joint(int(self.model.actuator_trnid[index, 0])).name for index in self.arm_actuator_ids],
            "state": "six measured joint angles [rad], then gripper closure from mean driver angle / 0.8; 0=open, 1=closed",
            "action": "six absolute joint targets [rad], then gripper closure command; 0=open, 1=closed",
            "transitions": "state/image/qpos at t, action/control_targets applied over [t,t+1), next_state/next_qpos at t+1",
            "images": {
                "enabled": self.renderer is not None,
                "resolution": [640, 480],
                "layout": "HWC RGB uint8",
                "paths": {"scene": "rgb_scene/{frame:06d}.png", "wrist": "rgb_wrist/{frame:06d}.png"} if self.renderer is not None else {},
            },
            "limitations": "Fixed scene baseline demonstrations; LeRobot conversion and scene randomization are separate steps.",
        }
        (self.directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ScriptedExpert:
    """按阶段执行真实接触抓取与插孔，通过实际 TCP 反馈补偿位置跟踪误差。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        recorder: EpisodeRecorder | None = None,
        window: TeleopWindow | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.recorder = recorder
        self.window = window
        self.controls = data.ctrl.copy()
        self.ik = KeyboardTeleop(model, data, self.controls)
        self.num_substeps = round(1.0 / (CONTROL_FREQUENCY_HZ * model.opt.timestep))
        self.dt = self.num_substeps * model.opt.timestep
        if not np.isclose(self.dt, 1.0 / CONTROL_FREQUENCY_HZ):
            raise ValueError("物理步长必须能够整除 20 Hz 控制周期")
        self.gripper_id = model.actuator("ur5e/gripper/fingers_actuator").id
        self.stage = "settle"
        self.steps = 0
        self.paused = False
        self.carrying = False
        self.grasp_offset = np.zeros(3)

    def _tick(self) -> None:
        start = time.monotonic()
        if self.window is not None:
            while True:
                self.window.poll_events()
                if not self.window.is_running():
                    raise RuntimeError("演示已由用户关闭，未保存为成功轨迹")
                for key in self.window.keys.pop_commands():
                    if key == ord("P"):
                        self.paused = not self.paused
                    elif key in (ord("1"), ord("2"), ord("3")):
                        camera = {ord("1"): "scene_camera", ord("2"): "ur5e/wrist_camera", ord("3"): "overview"}[key]
                        self.window.cam.fixedcamid = self.model.camera(camera).id
                if not self.paused:
                    break
                self.window.render(self.data, f"PAUSED | {self.stage}")
                time.sleep(0.02)
        if self.carrying:
            offset = self.data.site("ur5e/gripper/pinch").xpos - self.data.body("cube").xpos
            if np.linalg.norm(offset - self.grasp_offset) > 0.015:
                raise RuntimeError(f"阶段 {self.stage} 检测到方块滑落，未保存为成功轨迹")
        if self.recorder is not None:
            self.recorder.before_step(self.data, self.controls, self.stage)
        step_controls(self.model, self.data, self.controls, num_steps=self.num_substeps)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):
            raise RuntimeError(f"阶段 {self.stage} 出现非有限状态或 MuJoCo 警告")
        self.steps += 1
        if self.recorder is not None:
            self.recorder.after_step(self.data)
        if self.window is not None:
            self.window.render(self.data, f"SCRIPTED DEMO | {self.stage}\n{self.data.time:.2f} s")
            remaining = self.dt - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)

    def _hold(self, seconds: float) -> None:
        for _ in range(round(seconds / self.dt)):
            self._tick()

    def _reach(self, goal: np.ndarray, stage: str, *, speed: float = 0.06, tolerance: float = 0.0005) -> None:
        self.stage = stage
        print(f"阶段：{stage}", flush=True)
        for _ in range(round(15.0 / self.dt)):
            error = goal - self.data.site("ur5e/gripper/pinch").xpos
            if np.linalg.norm(error) < tolerance:
                self._hold(0.3)
                return
            # 基于实际 TCP 的外环反馈，消除仅跟踪理想 FK 时的重力下垂误差。
            delta = error * 4.0 * self.dt
            distance = np.linalg.norm(delta)
            if distance > speed * self.dt:
                delta *= speed * self.dt / distance
            message = self.ik.move_delta(self.data, self.controls, delta)
            if message:
                raise RuntimeError(f"阶段 {stage}：{message}")
            self._tick()
        raise RuntimeError(f"阶段 {stage} 超时，目标 {goal}，实际 {self.data.site('ur5e/gripper/pinch').xpos}")

    def _has_contact(self, first: str, second: str) -> bool:
        return any(
            {self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name} == {first, second}
            for contact in self.data.contact
        )

    def run(self) -> DemoResult:
        self._hold(0.5)
        source = self.data.body("cube").xpos.copy()
        target = self.data.site("socket_target").xpos.copy()
        self._reach(source + [0, 0, 0.15], "approach")
        # 夹持上部，给插槽孔壁留出夹指避让空间。
        self._reach(source + [0, 0, 0.035], "descend", speed=0.025)
        self.stage = "close_gripper"
        self.controls[self.gripper_id] = 255.0
        self._hold(2.0)
        self._reach(self.data.site("ur5e/gripper/pinch").xpos.copy() + [0, 0, 0.13], "lift")
        both_fingers = all(
            any(self._has_contact("cube_geom", f"ur5e/gripper/{side}_pad{number}") for number in (1, 2))
            for side in ("left", "right")
        )
        if self.data.body("cube").xpos[2] < source[2] + 0.06 or not both_fingers:
            raise RuntimeError("未检测到双指夹持并抬起方块，未保存为成功轨迹")
        self.grasp_offset = self.data.site("ur5e/gripper/pinch").xpos - self.data.body("cube").xpos
        self.carrying = True
        self._reach(target + self.grasp_offset + [0, 0, 0.13], "transfer")
        self._reach(target + self.grasp_offset + [0, 0, 0.04], "align", speed=0.025)
        if np.linalg.norm(self.data.body("cube").xpos[:2] - target[:2]) > 0.0015:
            raise RuntimeError("插孔前方块横向偏差超过 1.5 mm，未继续插入")
        self._reach(target + self.grasp_offset, "insert", speed=0.01, tolerance=0.0007)
        self.carrying = False
        self.stage = "release"
        self.controls[self.gripper_id] = 0.0
        self._hold(1.5)
        self._reach(self.data.site("ur5e/gripper/pinch").xpos.copy() + [0, 0, 0.13], "retreat", speed=0.04)
        self.stage = "verify"
        self._hold(1.0)
        error = self.data.body("cube").xpos - target
        cube_dof = self.model.joint("cube_free_joint").dofadr[0]
        stationary = np.linalg.norm(self.data.qvel[cube_dof:cube_dof + 6]) < 0.01
        success = bool(np.linalg.norm(error) < 0.002 and stationary and self._has_contact("cube_geom", "socket_floor"))
        if not success:
            raise RuntimeError(f"最终插入检查未通过：方块中心误差 {error}，静止状态 {stationary}")
        return DemoResult(success, float(self.data.time), self.steps, error.copy())


def run_demo(
    scene_path: Path,
    *,
    headless: bool = True,
    output_root: Path | None = None,
    record_images: bool = False,
    camera_name: str = "scene_camera",
) -> DemoResult:
    """执行示例，仅将成功完成的轨迹发布到输出目录。"""
    if record_images and output_root is None:
        raise ValueError("采集图像时必须指定输出目录")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ur5e/home").id)
    mujoco.mj_forward(model, data)
    with ExitStack() as stack:
        window = None
        if not headless:
            window = stack.enter_context(TeleopWindow(
                model, camera_name, title="UR5e Scripted Pick and Insert",
                help_text="P: pause/resume | 1: scene | 2: wrist | 3: overview | Esc: quit",
            ))
        renderer = None
        if record_images:
            renderer = stack.enter_context(mujoco.Renderer(model, width=640, height=480))
        recorder = None
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(stack.enter_context(TemporaryDirectory(prefix=".recording-", dir=output_root)))
            recorder = EpisodeRecorder(model, data, temporary, renderer)
        expert = ScriptedExpert(model, data, recorder=recorder, window=window)
        result = expert.run()
        if recorder is not None:
            recorder.save(result, data)
            episode_name = datetime.now(timezone.utc).strftime("episode_%Y%m%dT%H%M%S_%fZ")
            final_directory = output_root / episode_name
            recorder.directory.rename(final_directory)
            result.episode_dir = final_directory
        return result


def main() -> None:
    """启动窗口演示或无窗口采集，并打印成功检查与输出路径。"""
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无窗口运行，以最快速度推进物理仿真")
    parser.add_argument("--record-images", action="store_true", help="同步保存 20 Hz、640×480 的场景与腕部 RGB")
    parser.add_argument("--output", type=Path, default=project_dir / "outputs" / "scripted_demos")
    parser.add_argument("--camera", choices=("scene", "wrist", "overview"), default="scene")
    args = parser.parse_args()
    cameras = {"scene": "scene_camera", "wrist": "ur5e/wrist_camera", "overview": "overview"}
    result = run_demo(
        project_dir / "assets" / "scenes" / "workbench.xml",
        headless=args.headless, output_root=args.output, record_images=args.record_images,
        camera_name=cameras[args.camera],
    )
    print(f"演示成功：{result.duration_seconds:.2f} s，{result.steps} 步，方块中心误差 {np.linalg.norm(result.cube_error_meters) * 1000:.3f} mm")
    print(f"轨迹已保存：{result.episode_dir}")
    if not args.record_images:
        print("本次保存状态与动作；添加 --record-images 可同步采集双相机 RGB。")


if __name__ == "__main__":
    main()
