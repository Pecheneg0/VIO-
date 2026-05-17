"""ACO → CSV: разбор бинарного захвата в таблицу.

Схема одного кадра на проводе (индексы от начала кадра, j = смещение в файле):

    байты   | что это
    --------+--------------------------------------------------
    j+0..2  | сигнатура b\"ACO\" (3 байта)
    j+3     | L — сколько байт идёт *сразу после* этого байта (полезная нагрузка)
    j+4..   | полезная нагрузка длины L:  seq(1) + телеметрия + CRC(2)

То есть полный кадр занимает 3 + 1 + L = 4 + L байт от позиции j.

Телеметрия — фиксированный struct (см. STRUCT). Остальное в payload: 1 байт seq и 2 байта CRC,
поэтому длина блока телеметрии = L - 1 - 2 (при ожидаемом L это 96 байт).
"""

import csv
import struct
from pathlib import Path

SIG = b"ACO"

# После сигнатуры идёт один байт L, затем L байт payload
_PREFIX_LEN = len(SIG) + 1  # 4: ACO(3) + L(1)

# Внутри payload (длины в байтах)
_SEQ_LEN = 1
_CRC_LEN = 2

STRUCT = struct.Struct("<H2d12fih6f")

INPUT_BIN = Path("uart_raw.bin")
OUTPUT_CSV = Path("decoded2.csv")

HEADER = (
    "L;seq_hex;seq_dec;packetNumber;lat;lon;gps_height;hAcc;"
    "IMU_PITCH;IMU_ROLL;IMU_YAW;rateX;rateY;rateZ;ALTITUDE;presAltOffsetAtGround;"
    "ms5611_altitude;lidar_range;flags;state;IMU_ACCX;IMU_ACCY;IMU_ACCZ;"
    "IMU_MAGX;IMU_MAGY;IMU_MAGZ"
).split(";")


def main() -> None:
    data = INPUT_BIN.read_bytes()
    end = len(data)
    j = 0
    n_rows = 0

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter=";")
        w.writerow(HEADER)

        while j < end:
            j = data.find(SIG, j)
            if j < 0 or j + _PREFIX_LEN > end:
                break

            L = data[j + len(SIG)]
            # Слишком короткий payload не считаем кадром
            if L < 3:
                j += 1
                continue

            frame_len = _PREFIX_LEN + L
            if j + frame_len > end:
                break

            payload_start = j + _PREFIX_LEN
            seq = data[payload_start]

            telemetry_len = L - _SEQ_LEN - _CRC_LEN
            telemetry_start = payload_start + _SEQ_LEN
            body = data[telemetry_start : telemetry_start + telemetry_len]

            if len(body) != STRUCT.size:
                j += 1
                continue

            u = STRUCT.unpack(body)
            w.writerow(
                [
                    str(L),
                    f"0x{seq:02X}",
                    str(seq),
                    str(u[0]),
                    f"{u[1]:.10f}",
                    f"{u[2]:.10f}",
                    *[f"{x:.6f}" for x in u[3:15]],
                    str(u[15]),
                    str(u[16]),
                    *[f"{x:.6f}" for x in u[17:23]],
                ]
            )
            n_rows += 1
            j += frame_len

    print(f"{n_rows} строк → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

