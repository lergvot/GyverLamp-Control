import socket
import time

LAMP_IP = "192.168.1.143"  # IP лампы в локальной сети
LAMP_PORT = 8888  # Стандартный порт для управления лампой


def send_command(command, wait_response=True):
    try:
        # Отправляем пакет
        sent = sock.sendto(command.encode("utf-8"), (LAMP_IP, LAMP_PORT))
        expected = len(command.encode("utf-8"))
        if sent == expected:
            print(f"Отправлено: {command} ({sent} байт)")
        else:
            print(f"Ошибка: отправлено {sent} байт, ожидалось {expected}")

        # Ждем ответ
        if wait_response:
            # Установим таймаут в 2 секунды
            sock.settimeout(2.0)
            try:
                data, addr = sock.recvfrom(512)  # буфер размером 512 байта
                print(f"Получен ответ от {addr}: {data.decode('utf-8')}")
                return data.decode("utf-8")
            except socket.timeout:
                print("Таймаут: ответ не получен")
            except Exception as e:
                print(f"Ошибка при получении ответа: {e}")
            finally:
                # Сбросим таймаут
                sock.settimeout(None)

    except socket.error as e:
        print(f"Ошибка отправки: {e}")


try:
    # Создаём UDP-сокет
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Пример использования через функцию:
    send_command("P_ON")  # Включение лампы
    send_command("EFF 42")  # Установка эффекта 42
    send_command("BTN OFF")  # Выключение кнопки

    # Пример использования напрямую:
    # sock.sendto(b'EFF 5', ("192.168.1.143", 8888))

    # Примеры рисования точек
    send_command("DRAWON")  # Включение режима рисования
    send_command("CLR")  # Очистка экрана

    send_command("COL;0;255;0")  # Синий (R;B;G)
    send_command("DRW;8;8")  # Рисование точки по координатам
    send_command("COL;0;0;255")  # Зелёный (R;B;G)
    send_command("DRW;8;10")  # Рисование точки по координатам

finally:
    # Закрываем сокет
    sock.close()
