"""Example simulator command. Read one parameter vector and return its metrics."""

import json
import math
import sys


if __name__ == "__main__":
    theta = json.load(sys.stdin)
    amplitude, rate = theta["amplitude"], theta["rate"]
    print(json.dumps({
        "signal_t1": amplitude * math.exp(-rate),
        "signal_t2": amplitude * math.exp(-2 * rate),
    }))
