"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: from fusion_lab.workspace_support import get_tracking_params


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    # vi: TODO Part G — p_s = R @ p + t; loại tọa độ không hữu hạn.
    # vi: Camera cần x_s > 1e-6; FOV từ nội tại và bề rộng ảnh.
    # vi: Với cả lidar/camera: kiểm tra atan2(y_s, x_s) nằm trong sensor.fov.
    """Return True if state x is visible within the sensor horizontal field of view."""
    # 1. Trích xuất vị trí 3D từ state x (6x1 -> 3x1)
    p = np.asarray(x[:3]).reshape(3, 1)
    R = sensor.veh_to_sens[:3, :3]
    t = sensor.veh_to_sens[:3, 3:]
    p_s = R @ p + t

    if not np.all(np.isfinite(p_s)):
        return False

    coords = np.asarray(p_s).reshape(-1)
    x_s, y_s, z_s = float(coords[0]), float(coords[1]), float(coords[2])

    is_cam = hasattr(sensor, "f_i") or getattr(sensor, "type", None) == "camera"
    if is_cam and x_s <= 1e-6:
        return False

    # Tính góc quan sát ngang (horizontal azimuth)
    angle = np.arctan2(y_s, x_s)

    # Xử lý trường hợp sensor.fov là tuple hoặc số thực
    if isinstance(sensor.fov, (tuple, list)):
        if len(sensor.fov) == 2:
            min_fov, max_fov = sensor.fov
            # Nếu tuple dạng (min_angle, max_angle) như (-0.5, 0.5)
            if min_fov < 0 and max_fov > 0:
                if not (min_fov <= angle <= max_fov):
                    return False
            else:
                # Nếu tuple dạng (horizontal_fov, vertical_fov)
                if abs(angle) > min_fov / 2:
                    return False
        else:
            if abs(angle) > sensor.fov[0] / 2:
                return False
    else:
        # sensor.fov là số thực
        if abs(angle) > sensor.fov / 2:
            return False

    return True


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    # vi: TODO Part G — tính p_s = R @ p + t; trước phép chia kiểm tra hữu hạn
    # vi: và x_s > 1e-6, ngược lại raise ValueError có tọa độ.
    # vi: u = c_i - f_i * y_s/x_s; v = c_j - f_j * z_s/x_s.
    """Predict image-plane measurement h(x) using the pinhole camera model."""
    # 1. Trích xuất vị trí 3D từ state x (3x1)
    p = np.asarray(x[:3]).reshape(3, 1)
    R = sensor.veh_to_sens[:3, :3]
    t = sensor.veh_to_sens[:3, 3:]
    p_s = R @ p + t

    if not np.all(np.isfinite(p_s)):
        raise ValueError(f"Sensor coordinates are non-finite: {p_s}")

    coords = np.asarray(p_s).reshape(-1)
    x_s, y_s, z_s = float(coords[0]), float(coords[1]), float(coords[2])

    if x_s <= 1e-6:
        raise ValueError(f"Invalid camera depth x_s <= 1e-6: x_s={x_s}")

    # Mô hình Pinhole
    u = sensor.c_i - sensor.f_i * (y_s / x_s)
    v = sensor.c_j - sensor.f_j * (z_s / x_s)

    return np.asmatrix([[u], [v]])


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    # vi: TODO Part G — z mat 2x1; R diag sigma_cam_i^2, sigma_cam_j^2 từ params.
    z = np.matrix(z).reshape(-1, 1)
    params = get_tracking_params()
    R = np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2])
    return {"z": z, "R": R, "sensor": sensor}
