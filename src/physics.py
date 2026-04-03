"""Deterministic aircraft physics model."""

import math
from typing import Dict

GLIDE_SLOPE_DEG = 3.0
MAX_DESCENT_RATE_FPM = 3000
STD_DESCENT_RATE_FPM = 1500
TAXI_SPEED_KT = 20
TAXI_ACCEL_KT_PER_S = 5.0
TAXI_DECEL_KT_PER_S = 10.0
CRUISE_ALTITUDE_FT = 35000

KT_TO_FT_S = 6076.1 / 3600


def GlidePathUpdate(
    state: Dict[str, float],
    runway_threshold_x: float,
    runway_threshold_y: float,
    glide_slope_deg: float = 3.0,
) -> tuple:
    slope_rad = math.radians(glide_slope_deg)
    altitude_ft = float(state["altitude_ft"])
    horizontal_dist = altitude_ft / math.tan(slope_rad) if altitude_ft > 0 else 0.0
    x = runway_threshold_x + horizontal_dist
    y = runway_threshold_y
    return (x, y)


def DescentRateUpdate(
    current_altitude_ft: float,
    target_altitude_ft: float,
    dt_s: float,
) -> float:
    alt = float(current_altitude_ft)
    if alt <= 10000.0:
        rate_fpm = 500.0
    elif alt >= CRUISE_ALTITUDE_FT:
        rate_fpm = 1500.0
    else:
        t = (alt - 10000.0) / (CRUISE_ALTITUDE_FT - 10000.0)
        rate_fpm = 500.0 + 1000.0 * t
    descent_ft = rate_fpm * dt_s / 60.0
    new_alt = max(0.0, alt - descent_ft)
    return new_alt


def SurfaceMovementUpdate(
    current_x: float,
    current_y: float,
    heading_deg: float,
    speed_kt: float,
    dt_s: float,
) -> tuple:
    speed_kt = max(0.0, min(float(speed_kt), TAXI_SPEED_KT))
    heading_rad = math.radians(float(heading_deg))
    distance_ft = speed_kt * KT_TO_FT_S * dt_s
    x_new = current_x + distance_ft * math.sin(heading_rad)
    y_new = current_y + distance_ft * math.cos(heading_rad)
    return (x_new, y_new)


def HeadingUpdate(
    current_heading_deg: float,
    target_heading_deg: float,
    max_turn_rate_deg_per_s: float = 3.0,
    dt_s: float = 1.0,
) -> float:
    current = float(current_heading_deg) % 360.0
    target = float(target_heading_deg) % 360.0
    raw_delta = target - current
    delta = (
        raw_delta + 360.0
        if raw_delta < -180.0
        else raw_delta - 360.0
        if raw_delta > 180.0
        else raw_delta
    )
    max_turn = float(max_turn_rate_deg_per_s) * float(dt_s)
    if abs(delta) <= abs(max_turn):
        return target % 360.0
    return (current + math.copysign(max_turn, delta)) % 360.0


_WAKE_ORDER = ["CAT_A", "CAT_B", "CAT_C", "CAT_D"]

_WAKE_SPACING = {
    ("CAT_B", "CAT_A"): 3000,
    ("CAT_C", "CAT_A"): 3000,
    ("CAT_D", "CAT_A"): 3000,
    ("CAT_C", "CAT_B"): 5000,
    ("CAT_D", "CAT_B"): 5000,
    ("CAT_D", "CAT_C"): 6000,
}


def WakeCategorySpacing(
    leading_category: str,
    trailing_category: str,
) -> int:
    if leading_category not in _WAKE_ORDER or trailing_category not in _WAKE_ORDER:
        raise ValueError(
            f"Invalid wake category: {leading_category!r} or {trailing_category!r}"
        )
    if leading_category == trailing_category:
        return 5000
    key = (leading_category, trailing_category)
    if key in _WAKE_SPACING:
        return _WAKE_SPACING[key]
    return 5000
