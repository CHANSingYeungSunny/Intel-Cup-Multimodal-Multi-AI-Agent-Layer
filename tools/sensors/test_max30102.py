#!/usr/bin/env python3

import json
import time
import statistics
import sys
from smbus2 import SMBus


I2C_BUS = 1
MAX30102_ADDR = 0x57

REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01
REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CONFIG = 0x08
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D
REG_PART_ID = 0xFF


def read_reg(bus, reg):
    return bus.read_byte_data(MAX30102_ADDR, reg)


def write_reg(bus, reg, value):
    bus.write_byte_data(MAX30102_ADDR, reg, value)


def read_fifo_sample(bus):
    data = bus.read_i2c_block_data(MAX30102_ADDR, REG_FIFO_DATA, 6)

    red = ((data[0] << 16) | (data[1] << 8) | data[2]) & 0x3FFFF
    ir = ((data[3] << 16) | (data[4] << 8) | data[5]) & 0x3FFFF

    return red, ir


def setup_max30102(bus):
    # Reset
    write_reg(bus, REG_MODE_CONFIG, 0x40)
    time.sleep(0.2)

    # Clear interrupts
    try:
        read_reg(bus, REG_INTR_STATUS_1)
        read_reg(bus, REG_INTR_STATUS_2)
    except Exception:
        pass

    # Reset FIFO
    write_reg(bus, REG_FIFO_WR_PTR, 0x00)
    write_reg(bus, REG_OVF_COUNTER, 0x00)
    write_reg(bus, REG_FIFO_RD_PTR, 0x00)

    # FIFO sample average = 4
    write_reg(bus, REG_FIFO_CONFIG, 0x4F)

    # SpO2 mode: RED + IR
    write_reg(bus, REG_MODE_CONFIG, 0x03)

    # ADC range 4096 nA, sample rate 100 Hz, pulse width 411 us
    write_reg(bus, REG_SPO2_CONFIG, 0x27)

    # LED currents
    write_reg(bus, REG_LED1_PA, 0x24)
    write_reg(bus, REG_LED2_PA, 0x24)


def read_max30102():
    with SMBus(I2C_BUS) as bus:
        part_id = read_reg(bus, REG_PART_ID)

        setup_max30102(bus)

        red_samples = []
        ir_samples = []

        time.sleep(0.2)

        for _ in range(80):
            red, ir = read_fifo_sample(bus)
            red_samples.append(red)
            ir_samples.append(ir)
            time.sleep(0.02)

    red_avg = statistics.mean(red_samples)
    ir_avg = statistics.mean(ir_samples)

    red_delta = max(red_samples) - min(red_samples)
    ir_delta = max(ir_samples) - min(ir_samples)

    finger_present = ir_avg > 10000 and ir_delta > 50

    if finger_present:
        ppg_status = "finger_detected_raw_signal_ok"
    elif ir_avg > 10000:
        ppg_status = "light_detected_but_signal_not_pulsing"
    else:
        ppg_status = "no_finger_or_signal_too_weak"

    return {
        "max30102": {
            "detected": True,
            "address": "0x57",
            "part_id": f"0x{part_id:02X}",
            "red_raw": int(red_samples[-1]),
            "ir_raw": int(ir_samples[-1]),
            "red_avg": round(red_avg, 2),
            "ir_avg": round(ir_avg, 2),
            "red_delta": int(red_delta),
            "ir_delta": int(ir_delta),
            "finger_present": finger_present,
            "ppg_status": ppg_status,
            "bpm": None
        }
    }


def main():
    try:
        print(json.dumps(read_max30102(), indent=2))
    except Exception as e:
        print(
            json.dumps(
                {
                    "max30102": {
                        "detected": False,
                        "address": "0x57",
                        "error": str(e)
                    }
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
