import struct
import csv

# --- Настройки из твоих скриптов ---
INPUT_FILE = 'uart_raw.bin'
OUTPUT_CSV = 'decoded_data.csv'
SIG = b"ACO"
# Твоя структура: H(2)+2d(16)+12f(48)+i(4)+h(2)+6f(24) = 96 байт данных
STRUCT = struct.Struct("<H2d12fih6f")
DATA_SIZE = 96 
# Полный размер: SIG(3) + DATA(96) + CRC(1) = 100 байт (проверь это число!)
PACKET_SIZE = 3 + DATA_SIZE + 1 

HEADER = (
    "packetNumber;lat;lon;gps_height;hAcc;"
    "IMU_PITCH;IMU_ROLL;IMU_YAW;rateX;rateY;rateZ;ALTITUDE;presAltOffsetAtGround;"
    "ms5611_altitude;lidar_range;flags;state;IMU_ACCX;IMU_ACCY;IMU_ACCZ;"
    "IMU_MAGX;IMU_MAGY;IMU_MAGZ"
).split(";")

def decode_file():
    print(f"[INFO] Начинаю декодирование {INPUT_FILE}...")
    
    try:
        with open(INPUT_FILE, "rb") as f:
            raw_data = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Файл {INPUT_FILE} не найден.")
        return

    count = 0
    pos = 0
    
    with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, delimiter=';')
        writer.writerow(HEADER)

        # Ищем сигнатуру по всему файлу
        while pos <= len(raw_data) - PACKET_SIZE:
            # Ищем начало пакета "ACO"
            if raw_data[pos:pos+3] == SIG:
                packet = raw_data[pos : pos + PACKET_SIZE]
                
                # Извлекаем полезную нагрузку для распаковки (96 байт после SIG)
                payload = packet[3 : 3 + DATA_SIZE]
                
                # Проверка CRC (если в конце пакета реально 1 байт контрольной суммы)
                # Если в твоем протоколе нет CRC, просто закомментируй проверку
                received_crc = packet[-1]
                calculated_crc = sum(payload) % 256
                
                if received_crc == calculated_crc:
                    try:
                        decoded = STRUCT.unpack(payload)
                        writer.writerow(decoded)
                        count += 1
                        pos += PACKET_SIZE # Сдвигаемся на целый пакет
                        continue
                    except struct.error:
                        pass
                
            pos += 1 # Если не пакет, ищем со следующего байта

    print(f"[OK] Готово! Распознано пакетов: {count}")
    print(f"[INFO] Данные сохранены в {OUTPUT_CSV}")

if __name__ == "__main__":
    decode_file()

