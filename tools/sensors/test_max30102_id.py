#!/usr/bin/env python3

import json
import sys
from smbus2 import SMBus

I2C_BUS = 1
MAX30102_ADDR = 0x57
REG_PART_ID = 0xFF
REG_REV_ID = 0xFE

def main():
    try:
        with SMBus(I2C_BUS) as bus:
            part_id = bus.read_byte_data(MAX30102_ADDR, REG_PART_ID)
            rev_id = bus.read_byte_data(MAX30102_ADDR, REG_REV_ID)

        print(json.dumps({
            "max30102": {
                "detected": True,
                "address": "0x57",
                "part_id": f"0x{part_id:02X}",
                "revision_id": f"0x{rev_id:02X}"
            }
        }, indent=2))

    except Exception as e:
        print(json.dumps({
            "max30102": {
                "detected": False,
                "address": "0x57",
                "error": str(e)
            }
        }, indent=2), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
