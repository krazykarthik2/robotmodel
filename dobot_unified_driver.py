"""
Unified Dobot Magician Controller Supporting Dual Protocols:
1. Native Serial / Protocol Communication (pydobot) - works cross-platform with direct USB serial, queue unstick, & alarm clearing
2. Official Dobot DLL (DobotDll.dll) - official C++ DLL driver

Automatically selects whichever connection is available, and falls back to safe simulation dry-run when hardware is not plugged in.
"""
import os
import sys
import time
import struct
import numpy as np
from serial.tools import list_ports

# Check pydobot availability
try:
    import pydobot
    from pydobot.message import Message
    HAS_PYDOBOT = True
except ImportError:
    HAS_PYDOBOT = False

# DobotDll setup
DLL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dobot_dlls"))
if hasattr(os, "add_dll_directory") and os.path.exists(DLL_DIR):
    os.add_dll_directory(DLL_DIR)
sys.path.append(DLL_DIR)

try:
    import DobotDllType as dType
    HAS_DLL = True
except Exception:
    HAS_DLL = False

class DobotUnifiedDriver:
    def __init__(self, port=None, baudrate=115200, prefer_driver="auto"):
        self.port = port
        self.baudrate = baudrate
        self.prefer_driver = prefer_driver
        self.mode = None # 'pydobot', 'dll', or 'mock'
        self.device = None
        self.api = None
        self.connected = False

    def connect(self):
        """Attempts connection using pydobot first (most reliable direct USB protocol), then DLL."""
        # Detect active ports and rank them by hardware description
        all_ports = list(list_ports.comports())
        print(f"[INFO] Detected system serial ports:")
        for p in all_ports:
            print(f"       -> {p.device}: {p.description}")

        # Dobot/USB-to-Serial chips (CH9102, CH340, FTDI, CP210x, Dobot, Prolific)
        priority_keywords = ["ch9102", "ch340", "ch341", "usb-enhanced-serial", "cp210", "ftdi", "dobot", "usb-serial", "usb serial"]
        
        priority_ports = []
        regular_ports = []
        bluetooth_ports = []

        for p in all_ports:
            desc_lower = (p.description or "").lower()
            hwid_lower = (p.hwid or "").lower()
            
            if any(k in desc_lower or k in hwid_lower for k in priority_keywords):
                priority_ports.append(p.device)
            elif "bluetooth" in desc_lower:
                # Bluetooth serial ports often freeze on Windows when opening, put them last
                bluetooth_ports.append(p.device)
            else:
                regular_ports.append(p.device)

        sorted_ports = priority_ports + regular_ports + bluetooth_ports

        if self.port:
            # If explicit port passed, check it first, followed by others as fallback
            target_ports = [self.port] + [p for p in sorted_ports if p != self.port]
        else:
            target_ports = sorted_ports

        print(f"[INFO] Prioritized port search order: {target_ports}")

        # 1. Try PyDobot Native Serial Protocol (proven in working_python_version.py)
        if HAS_PYDOBOT and self.prefer_driver in ["auto", "pydobot"]:
            for p in target_ports:
                if not p:
                    continue
                try:
                    print(f"[INFO] Trying native serial connection on {p}...")
                    dev = pydobot.Dobot(port=p, verbose=False)
                    
                    # Fix queue and clear alarms (proven snippet from working_python_version.py)
                    self._init_pydobot_queue_and_alarms(dev)

                    pose = dev.pose()
                    self.device = dev
                    self.mode = "pydobot"
                    self.connected = True
                    print(f"[SUCCESS] Connected via native serial on {p}! Current pose: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}")
                    return True
                except Exception as e:
                    # Not a dobot on this port
                    pass

        # 2. Try Official DobotDll.dll
        if HAS_DLL and self.prefer_driver in ["auto", "dll"]:
            try:
                print(f"[INFO] Loading DobotDll.dll...")
                api = dType.load()
                dll_port = self.port if self.port else ""
                res = dType.ConnectDobot(api, dll_port, self.baudrate)
                if res[0] == dType.DobotConnect.DobotConnect_NoError:
                    self.api = api
                    self.mode = "dll"
                    self.connected = True
                    dType.SetQueuedCmdClear(self.api)
                    dType.SetQueuedCmdStartExec(self.api)
                    dType.SetPTPCoordinateParams(self.api, 150, 150, 150, 150, isQueued=0)
                    dType.SetPTPCommonParams(self.api, 100, 100, isQueued=0)
                    pose = dType.GetPose(self.api)
                    print(f"[SUCCESS] Connected via DobotDll on {res[1]}! Pose: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}")
                    return True
            except Exception as e:
                print(f"[INFO] DobotDll check: {e}")

        # 3. Fallback: Offline Mock / Dry-Run Mode
        self.mode = "mock"
        self.connected = False
        print("[NOTICE] Real Dobot hardware not connected or USB unplugged.")
        print("[NOTICE] Operating in SIMULATION / DRY-RUN mode.")
        return False

    def _init_pydobot_queue_and_alarms(self, dev):
        """Clears queue, starts queue execution, and clears hardware alarms (msg id 20, 240, 245)."""
        # Clear Queue (245)
        msg_clear = Message()
        msg_clear.id = 245
        msg_clear.ctrl = 0x01
        dev._send_command(msg_clear)
        time.sleep(0.05)

        # Start Queue (240)
        msg_start = Message()
        msg_start.id = 240
        msg_start.ctrl = 0x01
        dev._send_command(msg_start)
        time.sleep(0.05)

        # Clear Alarms (20)
        msg_alarm = Message()
        msg_alarm.id = 20
        msg_alarm.ctrl = 0x01
        msg_alarm.params = bytearray()
        dev._send_command(msg_alarm)
        time.sleep(0.1)

    def get_pose(self):
        """Returns physical coordinates [x, y, z, r, j1, j2, j3, j4]."""
        if self.mode == "pydobot" and self.device:
            p = self.device.pose()
            return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "r": float(p[3]),
                    "j1": float(p[4]), "j2": float(p[5]), "j3": float(p[6]), "j4": float(p[7])}
        elif self.mode == "dll" and self.api:
            p = dType.GetPose(self.api)
            return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "r": float(p[3]),
                    "j1": float(p[4]), "j2": float(p[5]), "j3": float(p[6]), "j4": float(p[7])}
        else:
            return {"x": 200.0, "y": 0.0, "z": 100.0, "r": 0.0, "j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0}

    def move_to_xyz(self, x_mm, y_mm, z_mm, r_deg=0.0, wait=True):
        """Moves robot to target Cartesian position in mm."""
        if self.mode == "pydobot" and self.device:
            self.device.move_to(x_mm, y_mm, z_mm, r_deg, wait=wait)
        elif self.mode == "dll" and self.api:
            cmd_idx = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x_mm, y_mm, z_mm, r_deg, isQueued=1)[0]
            if wait:
                self._wait_dll_cmd(cmd_idx)
        else:
            print(f"[DRY-RUN MOVE] X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm, R={r_deg:.1f}deg")

    def set_gripper(self, state, wait=True):
        """Controls gripper: 'open', 'close', or 'off'."""
        if self.mode == "pydobot" and self.device:
            enable = True if state in ["open", "close"] else False
            grip = True if state == "close" else False
            self.device.grip(enable and grip)
            if not enable:
                self.device.suck(False)
            if wait:
                time.sleep(0.2)
        elif self.mode == "dll" and self.api:
            if state == "close":
                cmd = dType.SetEndEffectorGripper(self.api, 1, 1, isQueued=1)[0]
            elif state == "open":
                cmd = dType.SetEndEffectorGripper(self.api, 1, 0, isQueued=1)[0]
            else:
                cmd = dType.SetEndEffectorGripper(self.api, 0, 0, isQueued=1)[0]
            if wait:
                self._wait_dll_cmd(cmd)
                time.sleep(0.2)
        else:
            print(f"[DRY-RUN GRIPPER] State set to: {state.upper()}")

    def _wait_dll_cmd(self, cmd_idx, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if dType.GetQueuedCmdCurrentIndex(self.api)[0] >= cmd_idx:
                break
            time.sleep(0.02)

    def close(self):
        if self.mode == "pydobot" and self.device:
            try:
                self.device.close()
            except Exception:
                pass
        elif self.mode == "dll" and self.api:
            try:
                dType.DisconnectDobot(self.api)
            except Exception:
                pass
        self.connected = False
        print("[INFO] Robot connection closed.")

if __name__ == "__main__":
    driver = DobotUnifiedDriver()
    driver.connect()
    pose = driver.get_pose()
    print("Initial Pose readout:", pose)
    driver.move_to_xyz(200, 0, 100)
    driver.close()
