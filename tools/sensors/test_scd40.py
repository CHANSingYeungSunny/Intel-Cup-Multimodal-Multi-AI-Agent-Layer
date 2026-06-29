#!/usr/bin/env python3

import json
import time
import sys
from smbus2 import SMBus, i2c_msg


I2C_BUS = 1
SCD40_ADDR = 0x62


CMD_START_PERIODIC = [0x21, 0xB1]
CMD_READ_MEASUREMENT = [0xEC, 0x05]
CMD_STOP_PERIODIC = [0x3F, 0x86]


def scd4x_crc8(data):
    """
    Sensirion CRC-8:
    polynomial 0x31, init 0xFF
    """
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def write_command(bus, command_bytes):
    msg = i2c_msg.write(SCD40_ADDR, command_bytes)
    bus.i2c_rdwr(msg)


def read_bytes(bus, length):
    msg = i2c_msg.read(SCD40_ADDR, length)
    bus.i2c_rdwr(msg)
    return list(msg)


def parse_word_with_crc(raw, index):
    msb = raw[index]
    lsb = raw[index + 1]
    crc = raw[index + 2]

    expected_crc = scd4x_crc8([msb, lsb])
    if crc != expected_crc:
        raise ValueError(
            f"CRC mismatch at bytes {index}-{index + 2}: "
            f"got 0x{crc:02X}, expected 0x{expected_crc:02X}"
        )

    return (msb << 8) | lsb


def read_scd40_once():
    with SMBus(I2C_BUS) as bus:
        # Stop any previous periodic measurement session.
        # This avoids "sensor already measuring" or stale state problems.
        try:
            write_command(bus, CMD_STOP_PERIODIC)
            time.sleep(0.5)
        except Exception:
            pass

        # Start periodic measurement.
        write_command(bus, CMD_START_PERIODIC)

        # SCD40 updates every ~5 seconds.
        time.sleep(5.2)

        # Request measurement.
        write_command(bus, CMD_READ_MEASUREMENT)
        time.sleep(0.02)

        raw = read_bytes(bus, 9)

        # Stop after one reading so the bus is left clean.
        try:
            write_command(bus, CMD_STOP_PERIODIC)
            time.sleep(0.5)
        except Exception:
            pass

    co2_raw = parse_word_with_crc(raw, 0)
    temp_raw = parse_word_with_crc(raw, 3)
    rh_raw = parse_word_with_crc(raw, 6)

    co2_ppm = co2_raw
    temperature_c = -45.0 + 175.0 * temp_raw / 65535.0
    humidity_percent = 100.0 * rh_raw / 65535.0

    return {
        "scd40": {
            "co2_ppm": int(co2_ppm),
            "temperature_c": round(temperature_c, 2),
            "humidity_percent": round(humidity_percent, 2),
        }
    }


def main():
    try:
        data = read_scd40_once()
        print(json.dumps(data, indent=2))
    except PermissionError:
        print(
            json.dumps(
                {
                    "error": "Permission denied. Try running with sudo or add your user to the i2c group."
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
