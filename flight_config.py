from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CameraConfig:
    tilt_deg: float = 30.0
    fov_deg: float = 120.0
    width: int = 640
    height: int = 480

@dataclass
class VisionConfig:
    rer_smoothing_alpha: float = 0.2
    num_flow_points: int = 300
    flow_gen_dist_min: float = 5.0
    flow_gen_dist_max: float = 150.0

@dataclass
class ControlConfig:
    k_yaw: float = 2.01
    k_pitch: float = 7.65

    # Cruise / Dive Logic
    dive_trigger_rer: float = 0.08
    dive_trigger_v_threshold: float = 0.42
    cruise_pitch_gain: float = 0.22

    # Pitch Bias Logic: bias = A + B * pitch
    pitch_bias_intercept: float = 0.40
    pitch_bias_slope: float = 0.33
    pitch_bias_min: float = -0.1
    pitch_bias_max: float = 0.3

    # Camera Tilt Compensation (v_target)
    # v_target = base + slope * (pitch + offset)
    v_target_slope_steep: float = 0.33
    v_target_slope_shallow: float = -0.11
    v_target_intercept: float = 0.5
    v_target_pitch_threshold: float = -1.2

    # Thrust
    thrust_base_intercept: float = 0.70
    thrust_base_slope: float = 0.38
    thrust_min: float = 0.15
    thrust_max: float = 0.5 # Base max

    rer_target: float = 0.25
    k_rer: float = 1.37

    flare_gain: float = 1.72
    flare_threshold_offset: float = 0.25

    # Final Mode (Disabled for now to simplify logic)
    final_mode_v_threshold_low: float = -10.0
    final_mode_v_threshold_high: float = 10.0

    final_mode_yaw_gain: float = 0.2
    final_mode_yaw_limit: float = 0.5

    final_mode_undershoot_pitch_target: float = 0.0
    final_mode_undershoot_pitch_gain: float = 4.0
    final_mode_undershoot_thrust_base: float = 0.55
    final_mode_undershoot_thrust_gain: float = 2.0

    final_mode_overshoot_pitch_target: float = -0.08
    final_mode_overshoot_pitch_gain: float = 4.0
    final_mode_overshoot_thrust_base: float = 0.55
    final_mode_overshoot_thrust_gain: float = 1.0
    final_mode_overshoot_v_target: float = 0.24

    # Velocity Estimation & Speed Limiting
    velocity_limit: float = 5.0
    braking_pitch_gain: float = 0.2
    max_braking_pitch_rate: float = 1.0
    velocity_smoothing_alpha: float = 0.5

@dataclass
class MissionConfig:
    target_alt: float = 50.0
    enable_staircase: bool = False
    staircase_drop: float = 20.0
    staircase_trigger_dist: float = 15.0
    staircase_trigger_alt: float = 30.0

@dataclass
class FlightConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'FlightConfig':
        config = FlightConfig()
        if 'camera' in data:
            for k, v in data['camera'].items():
                if hasattr(config.camera, k):
                    setattr(config.camera, k, v)
        if 'vision' in data:
            for k, v in data['vision'].items():
                if hasattr(config.vision, k):
                    setattr(config.vision, k, v)
        if 'control' in data:
            for k, v in data['control'].items():
                if hasattr(config.control, k):
                    setattr(config.control, k, v)
        if 'mission' in data:
            for k, v in data['mission'].items():
                if hasattr(config.mission, k):
                    setattr(config.mission, k, v)
        return config
