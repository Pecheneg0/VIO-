import time

import serial

# --- настрой под плату ---
DEV = "/dev/ttyS0"
BAUD = 921600
TIMEOUT = 0.05

# сколько секунд писать
RECORD_SEC = 100.0

# куда писать (рядом со скриптом)
OUT = "uart_raw.bin"


def main():
    t_end = time.monotonic() + RECORD_SEC
    n = 0
    t0 = time.time()

    with serial.Serial(DEV, BAUD, timeout=TIMEOUT) as port, open(OUT, "wb") as fp:
        while time.monotonic() < t_end:
            chunk = port.read(65536)
            if chunk:
                fp.write(chunk)
                n += len(chunk)

    dt = time.time() - t0
    print(OUT)
    print("bytes", n)
    print("sec", f"{dt:.3f}")
    if dt > 0:
        print("avg_Bps", int(n / dt))


if __name__ == "__main__":
    main()

