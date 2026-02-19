
import sys
import os
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.validate_dive_tracking import DiveValidator
from flight_config import FlightConfig

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("DebugVIOStability")

def main():
    print("--- Debugging VIO Stability (Scenario 4) ---")
    config = FlightConfig()

    # Scenario 4: Alt 60, Dist 80
    validator = DiveValidator(
        use_ground_truth=True,
        use_blind_mode=True,
        init_alt=60.0,
        init_dist=80.0,
        config=config
    )

    # Run short
    hist = validator.run(duration=5.0)

if __name__ == "__main__":
    main()
