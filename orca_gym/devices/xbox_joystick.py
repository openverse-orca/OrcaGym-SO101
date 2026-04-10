import pygame
import sys
import copy
import numpy as np

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class XboxJoystick:
    def __init__(self):
        self._joystick_state = None

    def update(self, joystick_state: dict):
        self._joystick_state = joystick_state.copy()

    def get_state(self):
        return self._joystick_state.copy()


    def capture_joystick_pos_ctrl(self) -> dict:
        move_x = self._joystick_state["axes"]["LeftStickX"]
        move_y = -self._joystick_state["axes"]["LeftStickY"]
        move_z = (1 + self._joystick_state["axes"]["RT"]) * 0.5 - (1 + self._joystick_state["axes"]["LT"]) * 0.5
        pos_ctrl = {'x': move_x, 'y': move_y, 'z': move_z}
        return pos_ctrl
    
    def capture_joystick_rot_ctrl(self) -> dict:
        yaw = self._joystick_state["axes"]["RightStickX"]
        pitch = self._joystick_state["axes"]["RightStickY"]
        roll = self._joystick_state["buttons"]["RB"] * 0.5 - self._joystick_state["buttons"]["LB"] * 0.5
        rot_ctrl = {'yaw': yaw, 'pitch': pitch, 'roll': roll}
        return rot_ctrl
    


class XboxJoystickManager:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # 创建display以确保事件队列工作
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        try:
            self._screen = pygame.display.set_mode((1, 1))
            pygame.display.set_caption("Xbox Controller")
            _logger.info("Pygame display 创建成功")
        except Exception as e:
            _logger.warning(f"无法创建 pygame display: {e}")
            self._screen = None
        
        # Initialize Joystick
        pygame.joystick.init()

        # Check the operating system
        if sys.platform.startswith('win'):
            self.platform = 'windows'
        elif sys.platform.startswith('linux'):
            self.platform = 'linux'
        else:
            raise Exception("Unsupported platform")

        self._pygame_joysticks = []
        self._xbox_joysticks = []
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise Exception("No joystick detected")
        
        _logger.info(f"检测到 {joystick_count} 个 Xbox 手柄")
        for i in range(joystick_count):
            self._pygame_joysticks.append(pygame.joystick.Joystick(i))
            self._pygame_joysticks[-1].init()
            xbox_joystick = XboxJoystick()
            self._xbox_joysticks.append(xbox_joystick)
            joystick_name = self._pygame_joysticks[-1].get_name()
            _logger.info(f"手柄 {i}: {joystick_name}")

        # Initialize joystick state
        init_joystick_state = {
            "buttons": {
                "A": 0, "B": 0, "X": 0, "Y": 0,
                "LB": 0, "RB": 0, "Back": 0, "Start": 0,
                "LeftStick": 0, "RightStick": 0
            },
            "axes": {
                "LeftStickX": 0.0, "LeftStickY": 0.0,
                "RightStickX": 0.0, "RightStickY": 0.0,
                "LT": -1.0, "RT": -1.0
            },
            "hats": {
                "DPad": (0, 0)
            }
        }

        self._joystick_states = [copy.deepcopy(init_joystick_state) for i in range(joystick_count)]

        self.BUTTON_MAP = {
            "windows":{
                0: "A", 1: "B", 2: "X", 3: "Y",
                4: "LB", 5: "RB", 6: "Back", 7: "Start",
                8: "LeftStick", 9: "RightStick"},
            "linux":{            
                0: "A", 1: "B", 2: "X", 3: "Y",
                4: "LB", 5: "RB", 6: "Back", 7: "Start",
                9: "LeftStick", 10: "RightStick"}  # Linux下 9=左摇杆, 10=右摇杆
        }

        self.AXIS_MAP = {
            "windows": {
                0: "LeftStickX", 1: "LeftStickY",
                2: "RightStickX", 3: "RightStickY",
                4: "LT", 5: "RT"},
            "linux": {
                0: "LeftStickX", 1: "LeftStickY",
                3: "RightStickX", 4: "RightStickY",
                2: "LT", 5: "RT"}
        }

        return
    
    def get_joystick_names(self):
        return [joystick.get_name() for joystick in self._pygame_joysticks]

    def get_joystick(self, joystick_name) -> XboxJoystick:
        for i, joystick in enumerate(self._pygame_joysticks):
            if joystick.get_name() == joystick_name:
                return self._xbox_joysticks[i]

        raise Exception("Joystick not found! name: ", joystick_name)
        
    def update(self):
        events = pygame.event.get()
        
        # 调试模式：显示按钮事件（临时开启以调试）
        DEBUG_XBOX_BUTTON = True
        
        for event in events:
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button in self.BUTTON_MAP[self.platform]:
                    button_name = self.BUTTON_MAP[self.platform][event.button]
                    self._joystick_states[event.joy]["buttons"][button_name] = 1
                    if DEBUG_XBOX_BUTTON:
                        print(f"[XBOX] 按下: {button_name} (ID={event.button})")
            if event.type == pygame.JOYBUTTONUP:
                if event.button in self.BUTTON_MAP[self.platform]:
                    button_name = self.BUTTON_MAP[self.platform][event.button]
                    self._joystick_states[event.joy]["buttons"][button_name] = 0
                    if DEBUG_XBOX_BUTTON:
                        print(f"[XBOX] 松开: {button_name} (ID={event.button})")
            if event.type == pygame.JOYAXISMOTION:
                if event.axis in self.AXIS_MAP[self.platform]:
                    self._joystick_states[event.joy]["axes"][self.AXIS_MAP[self.platform][event.axis]] = event.value
            if event.type == pygame.JOYHATMOTION:
                self._joystick_states[event.joy]["hats"]["DPad"] = event.value

        for i, joystick in enumerate(self._xbox_joysticks):
            joystick.update(self._joystick_states[i])

    def close(self):
        pygame.quit()
        _logger.info("Joysticks closed")
