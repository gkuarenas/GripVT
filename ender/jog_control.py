import serial
import time


class Ender3V2Controller:
    def __init__(self, port, baudrate=115200, response_queue=None):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.response_queue = response_queue

    def log(self, message, level="info"):
        if self.response_queue:
            self.response_queue.put(("log", (message, level)))

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.ser.flushInput()
            self.log(f"Connected on {self.port}.", "success")
            initial = self.ser.read(self.ser.in_waiting).decode(errors='ignore')
            for line in initial.splitlines():
                if line.strip():
                    self.log(f"PRINTER: {line.strip()}", "printer")
            return True
        except serial.SerialException as e:
            self.log(f"Connection error: {e}", "error")
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.log("Disconnected.", "info")

    def send_command(self, command, wait_for_ok=True):
        if not self.ser or not self.ser.is_open:
            return []
        self.log(f"SEND: {command}", "command")
        self.ser.write(f"{command}\n".encode('utf-8'))
        if not wait_for_ok:
            return []
        response_lines = []
        start = time.time()
        timeout = 30.0
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.log(f"RECV: {line}", "printer")
                        response_lines.append(line)
                        if "ok" in line.lower():
                            break
                except Exception:
                    pass
        return response_lines

    def move_axis(self, axis, distance, speed=3000):
        self.send_command("G91")
        self.send_command(f"G1 {axis.upper()}{distance:.3f} F{speed}")

    def home_all_axes(self):
        self.send_command("G28")

    def press_to_z(self, z_thresh_mm: float, feed: int = 300) -> None:
        self.send_command("G90")
        self.send_command(f"G1 Z{z_thresh_mm:.3f} F{feed}")
        self.send_command("M400")

    def retract_home(self, clearance_mm: float = 3.0, feed: int = 300) -> None:
        self.send_command("G90")
        self.send_command(f"G1 Z{clearance_mm:.3f} F{feed}")
        self.send_command("M400")

    def emergency_stop(self):
        self.send_command("M112", wait_for_ok=False)

    def disable_steppers(self):
        self.send_command("M84")

    def run_indentation_loop(
        self, z_mm: float, dwell_s: float, n_loops: int, stop_event
    ) -> None:
        for _ in range(n_loops):
            if stop_event.is_set():
                break
            self.move_axis('Z', -z_mm)
            if self.response_queue:
                self.response_queue.put(("pos_update", ("Z", -z_mm)))
            for _ in range(int(dwell_s * 10)):
                if stop_event.is_set():
                    break
                time.sleep(0.1)
            self.move_axis('Z', +z_mm)   # always return to zero
            if self.response_queue:
                self.response_queue.put(("pos_update", ("Z", +z_mm)))
            if stop_event.is_set():
                break
            time.sleep(0.3)
        self.log("Indentation loop complete.", "success")
