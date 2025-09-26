import socket
import time
import mss
import numpy as np
import colorsys
from typing import Tuple

# ==================== НАСТРОЙКИ КАЛИБРОВКИ ====================

# Настройки лампы
LAMP_IP = "192.168.1.143"
LAMP_PORT = 8888

# Выбор монитора
MONITOR_INDEX = 1  # 0 = все мониторы,1 = основной монитор, 2 = второй монитор и т.д.
SHOW_MONITOR_INFO = True  # Показать информацию о мониторах при запуске

# Калибровка яркости
BRI_MIN = 1  # Минимальная яркость (1-255)
BRI_MAX = 200  # Максимальная яркость (1-255)
BRI_GAMMA = 1.2  # Гамма-коррекция яркости (>1 - темнее, <1 - светлее)

# Калибровка насыщенности
SAT_MIN = 100  # Минимальная насыщенность (1-255)
SAT_MAX = 255  # Максимальная насыщенность (1-255)
SAT_BOOST = 1.35  # Усиление насыщенности (>1 - ярче цвета)

# Калибровка оттенка (Hue)
# Карта преобразования HSV Hue (0-360°) в SCA (0-100)
HUE_TO_SCA_MAP = [
    (0, 0),  # Красный (0°) -> 0
    (60, 20),  # Желтый (60°) -> 20
    (120, 39),  # Зеленый (120°) -> 39
    (180, 51),  # Голубой (180°) -> 51
    (240, 63),  # Синий (240°) -> 63
    (300, 85),  # Пурпурный (300°) -> 85
    (360, 100),  # Красный (360°) -> 100
]

# Настройки захвата экрана
UPDATE_INTERVAL = 0.1  # Интервал обновления (секунды)
SAMPLE_SIZE = 150  # Размер уменьшенного изображения для анализа (30-50 макс производительность, 100-200 хорошая точность, 300+ почти полное качество)
COLOR_CHANGE_THRESHOLD = 15  # Порог изменения цвета для обновления лампы (5-15 - высокая чувствительность, 20-40 - средняя чувствительность, 50+ - низкая чувствительность)

# Настройки плавности переходов
SMOOTH_TRANSITIONS = True  # Включить плавные переходы
TRANSITION_STEPS = 7  # Количество шагов для плавного перехода (миниум 2)

# Настройки отправки команд
COMMAND_BATCH_DELAY = 0.002  # Минимальная задержка между командами в батче (секунды)

# ==================== ОСНОВНОЙ КОД ====================


class LampSync:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sct = mss.mss()

        # Получаем информацию о мониторах
        self.monitors = self.sct.monitors
        if SHOW_MONITOR_INFO:
            self.print_monitor_info()

        # Выбираем монитор для захвата
        self.monitor = self.select_monitor(MONITOR_INDEX)
        self.last_rgb = (128, 128, 128)
        self.lamp_initialized = False

    def print_monitor_info(self):
        """Вывод информации о доступных мониторах"""
        print("=" * 50)
        print("ДОСТУПНЫЕ МОНИТОРЫ:")
        for i, monitor in enumerate(self.monitors):
            if i == 0:
                print(f"  {i}: Все мониторы (объединенная область)")
            else:
                print(
                    f"  {i}: Монитор {i} - {monitor['width']}x{monitor['height']} "
                    f"(левый верхний угол: {monitor['left']}, {monitor['top']})"
                )
        print("=" * 50)

    def select_monitor(self, index: int):
        """Выбор монитора для захвата"""
        if index < 0 or index >= len(self.monitors):
            print(f"Ошибка: монитор с индексом {index} не существует!")
            print(f"Используется монитор 1 по умолчанию.")
            return self.monitors[1]

        monitor = self.monitors[index]
        if index == 0:
            print(
                f"Выбрана объединенная область всех мониторов: {monitor['width']}x{monitor['height']}"
            )
        else:
            print(f"Выбран монитор {index}: {monitor['width']}x{monitor['height']}")

        return monitor

    def send_command_batch(self, commands: list, delay: float = COMMAND_BATCH_DELAY):
        """Отправка пакета команд без промежуточного обновления лампы"""
        try:
            for cmd in commands:
                sent = self.sock.sendto(cmd.encode("utf-8"), (LAMP_IP, LAMP_PORT))
                print(f"Отправлено: {cmd}")
                time.sleep(delay)  # Минимальная задержка для сетевого буфера
            return True
        except socket.error as e:
            print(f"Ошибка отправки команд: {e}")
            return False

    def send_single_command(self, command: str, wait_response: bool = False) -> bool:
        """Отправка одиночной команды (для инициализации)"""
        try:
            sent = self.sock.sendto(command.encode("utf-8"), (LAMP_IP, LAMP_PORT))
            print(f"Отправлено: {command}")

            if wait_response:
                self.sock.settimeout(1.0)
                try:
                    data, addr = self.sock.recvfrom(512)
                    print(f"Ответ: {data.decode('utf-8')}")
                except socket.timeout:
                    print("Таймаут ответа")
                finally:
                    self.sock.settimeout(None)

            return True
        except socket.error as e:
            print(f"Ошибка отправки: {e}")
            return False

    def initialize_lamp(self):
        """Инициализация лампы (только один раз при старте)"""
        if not self.lamp_initialized:
            print("Инициализация лампы...")
            # Включаем лампу и устанавливаем статический цвет
            self.send_single_command("P_ON")
            self.send_single_command("EFF 1")
            self.lamp_initialized = True
            time.sleep(0.1)  # Даем лампе время на инициализацию

    def get_average_screen_color_fast(self) -> Tuple[int, int, int]:
        """Быстрое получение среднего цвета экрана"""
        try:
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot)

            height, width = img.shape[:2]
            scale_factor = SAMPLE_SIZE / min(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            if new_width > 0 and new_height > 0:
                y_step = height // new_height
                x_step = width // new_width
                small_img = img[::y_step, ::x_step]
            else:
                small_img = img

            if small_img.shape[2] == 4:
                avg_b = np.mean(small_img[:, :, 0])
                avg_g = np.mean(small_img[:, :, 1])
                avg_r = np.mean(small_img[:, :, 2])
            else:
                avg_b = np.mean(small_img[:, :, 0])
                avg_g = np.mean(small_img[:, :, 1])
                avg_r = np.mean(small_img[:, :, 2])

            return int(avg_r), int(avg_g), int(avg_b)

        except Exception as e:
            print(f"Ошибка захвата экрана: {e}")
            return self.last_rgb

    def interpolate_hue_to_sca(self, hue_degrees: float) -> int:
        """Интерполяция оттенка HSV в значение SCA по карте"""
        for i in range(len(HUE_TO_SCA_MAP) - 1):
            h1, sca1 = HUE_TO_SCA_MAP[i]
            h2, sca2 = HUE_TO_SCA_MAP[i + 1]

            if h1 <= hue_degrees <= h2:
                ratio = (hue_degrees - h1) / (h2 - h1)
                return int(sca1 + (sca2 - sca1) * ratio)

        return HUE_TO_SCA_MAP[0][1]

    def apply_gamma_correction(self, value: float, gamma: float) -> float:
        """Применение гамма-коррекции к значению"""
        return value**gamma

    def rgb_to_lamp_params(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        """Преобразование RGB в параметры лампы с калибровкой"""
        # Нормализуем RGB
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0

        # Преобразуем в HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)

        # Яркость с калибровкой
        v_corrected = self.apply_gamma_correction(v, BRI_GAMMA)
        brightness = int(BRI_MIN + v_corrected * (BRI_MAX - BRI_MIN))
        brightness = max(1, min(255, brightness))

        # Насыщенность с калибровкой
        s_boosted = min(1.0, s * SAT_BOOST)
        saturation = int(SAT_MIN + s_boosted * (SAT_MAX - SAT_MIN))
        saturation = max(1, min(255, saturation))

        # Оттенок с использованием карты преобразования
        hue_degrees = h * 360
        lamp_hue = self.interpolate_hue_to_sca(hue_degrees)
        lamp_hue = max(0, min(100, lamp_hue))

        return brightness, saturation, lamp_hue

    def smooth_transition(
        self, start_rgb: Tuple[int, int, int], target_rgb: Tuple[int, int, int]
    ) -> list:
        """Создание плавного перехода между цветами"""
        transitions = []
        for i in range(TRANSITION_STEPS):
            ratio = i / (TRANSITION_STEPS - 1)
            r = int(start_rgb[0] + (target_rgb[0] - start_rgb[0]) * ratio)
            g = int(start_rgb[1] + (target_rgb[1] - start_rgb[1]) * ratio)
            b = int(start_rgb[2] + (target_rgb[2] - start_rgb[2]) * ratio)
            transitions.append((r, g, b))
        return transitions

    def set_lamp_color(self, r: int, g: int, b: int):
        """Установка цвета лампы - ВСЕ параметры отправляются сразу"""
        brightness, saturation, hue = self.rgb_to_lamp_params(r, g, b)

        print(
            f"RGB({r}, {g}, {b}) -> Лампa: BRI:{brightness} SPD:{saturation} SCA:{hue}"
        )

        # Формируем пакет команд для ОДНОВРЕМЕННОЙ отправки
        commands = [
            f"BRI {brightness}",  # Яркость
            f"SPD {saturation}",  # Насыщенность
            f"SCA {hue}",  # Оттенок
        ]

        # Отправляем все команды БЫСТРОЙ последовательностью
        self.send_command_batch(commands)

    def run_sync(self):
        """Основной цикл синхронизации"""
        try:
            print(f"Запуск синхронизации с монитором {MONITOR_INDEX}...")
            print("Для остановки нажмите Ctrl+C")

            # Инициализируем лампу один раз
            self.initialize_lamp()

            frame_count = 0
            last_update_time = time.time()

            while True:
                try:
                    # Получаем средний цвет экрана
                    r, g, b = self.get_average_screen_color_fast()

                    # Проверяем, достаточно ли цвет изменился
                    color_diff = (
                        abs(r - self.last_rgb[0])
                        + abs(g - self.last_rgb[1])
                        + abs(b - self.last_rgb[2])
                    )

                    current_time = time.time()
                    time_since_last_update = current_time - last_update_time

                    # Обновляем цвет если:
                    # 1. Цвет значительно изменился И
                    # 2. Прошло достаточно времени с последнего обновления
                    if (
                        color_diff > COLOR_CHANGE_THRESHOLD
                        and time_since_last_update >= UPDATE_INTERVAL
                    ):

                        if SMOOTH_TRANSITIONS and frame_count > 0:
                            # Плавный переход между цветами
                            transitions = self.smooth_transition(
                                self.last_rgb, (r, g, b)
                            )
                            for transition_rgb in transitions:
                                self.set_lamp_color(*transition_rgb)
                                time.sleep(UPDATE_INTERVAL / TRANSITION_STEPS)
                        else:
                            # Мгновенное изменение цвета
                            self.set_lamp_color(r, g, b)

                        self.last_rgb = (r, g, b)
                        last_update_time = current_time

                    frame_count += 1
                    time.sleep(0.05)  # Короткая задержка для снижения нагрузки на CPU

                except Exception as e:
                    print(f"Ошибка в основном цикле: {e}")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\nОстановка синхронизации...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Очистка ресурсов"""
        self.sct.close()
        self.sock.close()


if __name__ == "__main__":
    sync = LampSync()
    sync.run_sync()
