"""独立遥操窗口，接管键盘事件，保持 MuJoCo 显示选项不受遥操按键影响。"""

from collections import deque

import glfw
import mujoco


MOVEMENT_KEYS = frozenset(ord(key) for key in "WSADIJQE")
COMMAND_KEYS = frozenset(ord(key) for key in " FPR123")


class KeyboardState:
    """保存实际按住的键；切换命令只在按下边沿产生一次。"""

    def __init__(self) -> None:
        self.pressed: set[int] = set()
        self.commands: deque[int] = deque()

    def handle(self, key: int, action: int) -> None:
        """处理 PRESS/RELEASE，忽略操作系统的 REPEAT。"""
        if action == glfw.RELEASE:
            self.pressed.discard(key)
        elif action == glfw.PRESS and key not in self.pressed:
            self.pressed.add(key)
            if key in COMMAND_KEYS:
                self.commands.append(key)

    def clear(self) -> None:
        """失焦、暂停或复位时清除按键，避免失去松开事件后继续运动。"""
        self.pressed.clear()
        self.commands.clear()

    def pop_commands(self) -> list[int]:
        commands = list(self.commands)
        self.commands.clear()
        return commands


class TeleopWindow:
    """直接使用 GLFW 与 MuJoCo 渲染，按键不进入原生查看器快捷键处理。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str,
        *,
        title: str = "UR5e Keyboard Teleoperation",
        help_text: str | None = None,
    ) -> None:
        self.model = model
        self.keys = KeyboardState()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = model.camera(camera_name).id
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.window = None
        self.context = None
        self.help_text = help_text or (
            "W/S A/D: move | I/J: up/down | Q/E: yaw\n"
            "Space: gripper | F: fine | P: pause | R: reset\n"
            "1: scene | 2: wrist | 3: overview | Esc: quit"
        )
        if not glfw.init():
            raise RuntimeError("无法初始化 GLFW，请在可显示图形的桌面会话中运行遥操脚本")
        try:
            self.window = glfw.create_window(960, 720, title, None, None)
            if not self.window:
                raise RuntimeError("无法创建遥操图形窗口")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            glfw.set_key_callback(self.window, self.on_key)
            glfw.set_window_focus_callback(self.window, self.on_focus)
            glfw.set_window_iconify_callback(self.window, self.on_iconify)
        except Exception:
            self.close()
            raise

    def on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """所有键盘事件仅用于遥操，不修改光照、几何组或渲染标志。"""
        self.keys.handle(key, action)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def on_focus(self, window, focused: int) -> None:
        if not focused:
            self.keys.clear()

    def on_iconify(self, window, iconified: int) -> None:
        if iconified:
            self.keys.clear()

    def poll_events(self) -> None:
        glfw.poll_events()
        if not glfw.get_window_attrib(self.window, glfw.FOCUSED):
            self.keys.clear()

    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def render(self, data: mujoco.MjData, status: str) -> None:
        """渲染固定相机与遥操提示；显示状态在整个操作中保持一致。"""
        width, height = glfw.get_framebuffer_size(self.window)
        if width == 0 or height == 0:
            return
        # 图像采集可能切换到离屏上下文；渲染窗口前恢复自己的上下文。
        glfw.make_context_current(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model, data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport, status, "", self.context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport, self.help_text,
            "", self.context,
        )
        glfw.swap_buffers(self.window)

    def close(self) -> None:
        if self.context is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            self.context = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
