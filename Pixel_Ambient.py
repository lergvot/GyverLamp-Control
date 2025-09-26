"""
Pixel_Ambient — версия с попиксельной отсылкой команд на лампу.
"""

import os
import queue
import signal
import socket
import threading
import time
from typing import Dict, Optional, Tuple

import mss
import numpy as np

# ==================== НАСТРОЙКИ ====================

# === Сеть и лампа ===
LAMP_IP = "192.168.1.143"
LAMP_PORT = 8888
SOCKET_TIMEOUT = 1.0  # секунда

# === Разрешение сетки лампы ===
GRID_SIZE = 16  # 16x16

# === Монитор и частоты ===
MONITOR_INDEX = 1  # индекс монитора в mss.monitors (1..n); 0 - все мониторы
CAPTURE_FPS = 1  # сколько раз в секунду делать захват экрана
SEND_FPS = 30  # как часто отправлять изменения пикселей

# === Коррекция цвета и фильтры ===
GAMMA_CORRECTION = 1.5
SATURATION_BOOST = 1.5
MIN_BRIGHTNESS = 15
MAX_BRIGHTNESS = 225
USE_MEDIAN_INSTEAD_AVG = False

# === Отправка / производительность ===
MAX_WORKERS = 1  # воркеров отправки UDP
COMMAND_DELAY = 0.001  # задержка между отправками
INIT_COMMAND_DELAY = 0.1  # задержка при инициализации лампы
BATCH_SIZE = 1
LOG_CHANGES_ONLY = True
FLIP_VERTICAL = True

# === Отладка ===
ENABLE_TERMINAL_DISPLAY = True

# ==================== ТИПЫ ====================
Color = Tuple[int, int, int]
GridColors = Dict[Tuple[int, int], Color]


# ==================== AsyncLampController ====================
class AsyncLampController:
    """Асинхронная отправка команд на лампу через UDP с worker'ами и frame_id."""

    def __init__(self, ip: str, port: int, workers: int = MAX_WORKERS):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(SOCKET_TIMEOUT)

        self.task_queue: "queue.Queue" = queue.Queue()
        self._running = threading.Event()
        self._running.set()

        self._workers: list[threading.Thread] = []
        self._workers_count = max(1, workers)

        self._last_frame_id = 0
        self._frame_lock = threading.Lock()

        self._start_workers()

    def _start_workers(self) -> None:
        for i in range(self._workers_count):
            t = threading.Thread(
                target=self._worker_loop, name=f"LampWorker-{i}", daemon=True
            )
            t.start()
            self._workers.append(t)
        print(f"✅ Запущено {len(self._workers)} воркеров отправки")

    def _worker_loop(self) -> None:
        while self._running.is_set():
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if not task:
                    continue
                kind = task[0]

                if kind == "cmd":
                    cmd = task[1]
                    self._send_raw(cmd)

                elif kind == "pixel":
                    _, fid, x, y, r, g, b = task
                    with self._frame_lock:
                        if fid < self._last_frame_id:
                            # устарело — пропускаем
                            self.task_queue.task_done()
                            continue
                    self._send_pixel_now(x, y, r, g, b)

            except Exception as exc:
                print(f"❌ Ошибка в воркере при обработке {task}: {exc}")
            finally:
                self.task_queue.task_done()
                time.sleep(COMMAND_DELAY)

    def _send_raw(self, command: str) -> bool:
        try:
            self.sock.sendto(command.encode("utf-8"), (self.ip, self.port))
            return True
        except Exception as exc:
            print(f"❌ Ошибка при отправке '{command}': {exc}")
            return False

    def _send_pixel_now(self, x: int, y: int, r: int, g: int, b: int) -> bool:
        if FLIP_VERTICAL:
            y = GRID_SIZE - 1 - y
        if not self._send_raw(f"COL;{r};{b};{g}"):
            return False
        if not self._send_raw(f"DRW;{x};{y}"):
            return False
        return True

    def set_pixel_async(
        self, x: int, y: int, r: int, g: int, b: int, frame_id: int
    ) -> None:
        with self._frame_lock:
            if frame_id > self._last_frame_id:
                self._last_frame_id = frame_id
        self.task_queue.put(("pixel", frame_id, x, y, r, g, b))

    def send_command(self, command: str) -> None:
        self.task_queue.put(("cmd", command))

    def wait_for_all(self, timeout: Optional[float] = None) -> None:
        if timeout is None:
            self.task_queue.join()
            return
        end = time.time() + timeout
        while time.time() < end and not self.task_queue.empty():
            time.sleep(0.02)

    def stop(self) -> None:
        self._running.clear()
        for w in self._workers:
            w.join(timeout=1.0)
        try:
            self.sock.close()
        except Exception:
            pass
        print("✅ Воркеры остановлены")


# ==================== EnhancedScreenCapture ====================
class EnhancedScreenCapture:
    """Захват экрана с коррекцией цвета."""

    def __init__(self, monitor_index: int = MONITOR_INDEX, grid_size: int = GRID_SIZE):
        self.monitor_index = monitor_index
        self.grid_size = grid_size
        self.gamma_table = self._make_gamma_table(GAMMA_CORRECTION)

        try:
            with mss.mss() as sct:
                monitors = sct.monitors
            if self.monitor_index >= len(monitors):
                idx = 1 if len(monitors) > 1 else 0
                print(f"⚠️ MONITOR_INDEX {self.monitor_index} недоступен, выбран {idx}")
                self.monitor = monitors[idx]
            else:
                self.monitor = monitors[self.monitor_index]
        except Exception as exc:
            print(f"❌ Ошибка при получении списка мониторов: {exc}")
            self.monitor = {"left": 0, "top": 0, "width": 800, "height": 600}

    @staticmethod
    def _make_gamma_table(gamma: float) -> np.ndarray:
        return np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
        )

    def _apply_color_correction(self, r: int, g: int, b: int) -> Color:
        r_g = int(self.gamma_table[r])
        g_g = int(self.gamma_table[g])
        b_g = int(self.gamma_table[b])

        if SATURATION_BOOST != 1.0:
            h, s, v = self._rgb_to_hsv(r_g, g_g, b_g)
            s = min(1.0, s * SATURATION_BOOST)
            r_g, g_g, b_g = self._hsv_to_rgb(h, s, v)

        brightness = 0.299 * r_g + 0.587 * g_g + 0.114 * b_g
        if brightness < MIN_BRIGHTNESS:
            return 0, 0, 0
        if brightness > MAX_BRIGHTNESS:
            factor = MAX_BRIGHTNESS / brightness
            r_g = int(r_g * factor)
            g_g = int(g_g * factor)
            b_g = int(b_g * factor)

        return r_g, g_g, b_g

    @staticmethod
    def _rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
        r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
        cmax, cmin = max(r_n, g_n, b_n), min(r_n, g_n, b_n)
        d = cmax - cmin
        if d == 0:
            h = 0.0
        elif cmax == r_n:
            h = 60 * (((g_n - b_n) / d) % 6)
        elif cmax == g_n:
            h = 60 * (((b_n - r_n) / d) + 2)
        else:
            h = 60 * (((r_n - g_n) / d) + 4)
        s = d / cmax if cmax != 0 else 0.0
        v = cmax
        return h, s, v

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        if s == 0:
            val = int(v * 255)
            return val, val, val
        h = h % 360
        sector = h / 60
        i = int(sector)
        f = sector - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        if i == 0:
            rgb = (v, t, p)
        elif i == 1:
            rgb = (q, v, p)
        elif i == 2:
            rgb = (p, v, t)
        elif i == 3:
            rgb = (p, q, v)
        elif i == 4:
            rgb = (t, p, v)
        else:
            rgb = (v, p, q)
        return tuple(int(c * 255) for c in rgb)

    def get_grid_colors(self) -> GridColors:
        try:
            bbox = {
                "left": self.monitor.get("left", 0),
                "top": self.monitor.get("top", 0),
                "width": self.monitor.get("width", 800),
                "height": self.monitor.get("height", 600),
            }
            with mss.mss() as sct:
                shot = sct.grab(bbox)
                img = np.array(shot)

            if img.size == 0:
                return {}

            bgr = img[..., :3]
            rgb = bgr[..., ::-1]

            height, width = rgb.shape[:2]
            cell_h = height / self.grid_size
            cell_w = width / self.grid_size

            colors: GridColors = {}
            for gy in range(self.grid_size):
                y0 = int(round(gy * cell_h))
                y1 = (
                    int(round((gy + 1) * cell_h)) if gy < self.grid_size - 1 else height
                )
                for gx in range(self.grid_size):
                    x0 = int(round(gx * cell_w))
                    x1 = (
                        int(round((gx + 1) * cell_w))
                        if gx < self.grid_size - 1
                        else width
                    )
                    block = rgb[y0:y1, x0:x1]
                    if block.size == 0:
                        colors[(gx, gy)] = (0, 0, 0)
                        continue
                    if USE_MEDIAN_INSTEAD_AVG:
                        r = int(np.median(block[..., 0]))
                        g = int(np.median(block[..., 1]))
                        b = int(np.median(block[..., 2]))
                    else:
                        r = int(np.mean(block[..., 0]))
                        g = int(np.mean(block[..., 1]))
                        b = int(np.mean(block[..., 2]))
                    colors[(gx, gy)] = self._apply_color_correction(r, g, b)

            return colors
        except Exception as exc:
            print(f"❌ Ошибка capture: {exc}")
            return {}


# ==================== TerminalDisplay ====================
class TerminalDisplay:
    @staticmethod
    def rgb_to_ansi(r: int, g: int, b: int) -> str:
        return f"\033[48;2;{r};{g};{b}m"

    def display_grid(
        self, colors: GridColors, frame: int, changed_count: int, fps_target: int
    ) -> None:
        if not ENABLE_TERMINAL_DISPLAY:
            return
        os.system("cls" if os.name == "nt" else "clear")
        print("=== 🎨 PIXEL AMBIENT ===")
        print(f"Кадр: {frame} | Изменено: {changed_count} | SEND_FPS: {fps_target}")
        print(f"Монитор: {MONITOR_INDEX} | Сетка: {GRID_SIZE}x{GRID_SIZE}")
        print("-" * 50)
        grid = [[(0, 0, 0) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for (x, y), c in colors.items():
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                grid[y][x] = c
        for y in range(GRID_SIZE):
            line = ""
            for x in range(GRID_SIZE):
                r, g, b = grid[y][x]
                if r == g == b == 0:
                    line += "  "
                else:
                    line += f"{self.rgb_to_ansi(r, g, b)}  \033[0m"
            print(line)
        active = sum(1 for c in colors.values() if c != (0, 0, 0))
        print("-" * 50)
        print(f"Активных пикселей: {active}/{GRID_SIZE*GRID_SIZE}")
        print("Ctrl+C для выхода")
        print("-" * 50)


# ==================== PixelAmbient ====================
class PixelAmbient:
    def __init__(self):
        self.capture = EnhancedScreenCapture(MONITOR_INDEX, GRID_SIZE)
        self.lamp = AsyncLampController(LAMP_IP, LAMP_PORT, workers=MAX_WORKERS)
        self.display = TerminalDisplay()
        self._shared_colors: GridColors = {}
        self._shared_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self.frame_counter = 0

    def _signal_handler(self, signum, frame):
        print("\n🛑 Получен сигнал завершения, останавливаемся...")
        self._stop_event.set()

    def start(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

        if not self._init_lamp():
            print("❌ Не удалось инициализировать лампу, выходим")
            return

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="ScreenUpdater", daemon=True
        )
        self._capture_thread.start()
        print("✅ Поток захвата запущен")

        try:
            self._sender_loop()
        finally:
            self.stop()

    def stop(self):
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        self.lamp.wait_for_all(timeout=1.0)
        self.lamp.stop()
        print("✅ Приложение корректно завершено")

    def _init_lamp(self) -> bool:
        print("🔧 Инициализация лампы...")
        try:
            init_cmds = ["P_ON", "CLR", "DRAWON"]
            for c in init_cmds:
                self.lamp.send_command(c)
                time.sleep(INIT_COMMAND_DELAY)
            print("✅ Лампа инициализирована")
            return True
        except Exception as exc:
            print(f"❌ Init failed: {exc}")
            return False

    def _capture_loop(self):
        interval = 1.0 / CAPTURE_FPS
        while not self._stop_event.is_set():
            start = time.time()
            colors = self.capture.get_grid_colors()
            if colors:
                with self._shared_lock:
                    self._shared_colors = colors
            elapsed = time.time() - start
            to_sleep = interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def _sender_loop(self):
        send_interval = 1.0 / SEND_FPS
        last_sent_colors: GridColors = {}
        while not self._stop_event.is_set():
            start = time.time()
            with self._shared_lock:
                current = dict(self._shared_colors)

            if not current:
                time.sleep(send_interval)
                continue

            changed = []
            for coord, color in current.items():
                if coord not in last_sent_colors or last_sent_colors[coord] != color:
                    changed.append((coord, color))

            changed_count = 0
            if changed:
                self.frame_counter += 1
                frame_id = self.frame_counter
                for (x, y), (r, g, b) in changed:
                    self.lamp.set_pixel_async(x, y, r, g, b, frame_id=frame_id)
                    changed_count += 1
                if changed_count > 0 or not LOG_CHANGES_ONLY:
                    self.display.display_grid(
                        current, frame_id, changed_count, SEND_FPS
                    )

            last_sent_colors = current

            elapsed = time.time() - start
            to_sleep = send_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def print_settings(self) -> None:
        print("=== ТЕКУЩИЕ НАСТРОЙКИ ===")
        print(f"Сеть: {LAMP_IP}:{LAMP_PORT}")
        print(f"Монитор: {MONITOR_INDEX}, Сетка: {GRID_SIZE}x{GRID_SIZE}")
        print(f"CAPTURE_FPS: {CAPTURE_FPS}, SEND_FPS: {SEND_FPS}")
        print(f"Workers: {MAX_WORKERS}, COMMAND_DELAY: {COMMAND_DELAY}")
        print(f"Gamma: {GAMMA_CORRECTION}, SatBoost: {SATURATION_BOOST}")
        print("=" * 40)


def main() -> None:
    app = PixelAmbient()
    app.print_settings()
    app.start()


if __name__ == "__main__":
    main()
