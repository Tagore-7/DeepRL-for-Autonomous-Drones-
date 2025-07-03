from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RobustCfg:
    enabled: bool = False
    noise_factor: str = (
        "action"  # name of noise factor, such as state, action, reward, cost, robust_force (dynamics), robust_shape (mass), dynamics
    )
    noise_type: str = "gauss"  # name of the noise type, e.g., gauss, shift, uniform, Non_stationary
    noise_mu: float = 0.0  # noise mean (default: 0.0)
    noise_sigma: float = 0.0  # noise variance (default: 0.05)

    delay_steps: int = 0
    rotor_failures: str = "none"  # example: "0:10" or "2:20" to test rotor 0 failing for 10 steps, or rotor 2 failing for 20 steps, etc
    sign_flips: str = "none"  # example: "0, 2" to test r0 and r2 sign flips together, "0,2;1,3" to test r0,r2 together, then r1,r3 together
