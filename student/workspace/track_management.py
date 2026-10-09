"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

# vi: from fusion_lab.workspace_support import get_tracking_params
# vi: import numpy as np
import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    # vi: TODO Part H — đổi meas.z sang vehicle frame; x = [pos; 0 velocity];
    # vi: P block pos từ R xoay, vel từ sigma_p44/55/66; score = 1/window; state initialized.
    params = get_tracking_params()

    # 1. Chuyển meas.z từ hệ cảm biến sang hệ xe (sensor -> vehicle frame)
    # Tọa độ vị trí meas.z (3x1)
    z_s = np.asarray(meas.z[:3]).reshape(3, 1)
    # veh_to_sens = [R, t], sens_to_veh = [R.T, -R.T @ t]
    R_sens = meas.sensor.veh_to_sens[:3, :3]
    t_sens = meas.sensor.veh_to_sens[:3, 3:]
    pos_veh = R_sens.T @ (z_s - t_sens)

    # 2. Khởi tạo vector trạng thái x (6x1): [pos_x, pos_y, pos_z, 0, 0, 0]
    x = np.zeros((6, 1))
    x[:3] = pos_veh

    # 3. Khởi tạo ma trận hiệp phương sai P (6x6)
    # Ma trận hiệp phương sai đo lường xoay sang hệ xe: R_sens.T @ meas.R @ R_sens
    P = np.zeros((6, 6))
    P[:3, :3] = R_sens.T @ meas.R @ R_sens
    P[3, 3] = params.sigma_p44 ** 2
    P[4, 4] = params.sigma_p55 ** 2
    P[5, 5] = params.sigma_p66 ** 2

    # 4. Điểm tồn tại ban đầu và trạng thái
    score = 1.0 / params.window
    state = "initialized"

    return {
        "x": np.asmatrix(x),
        "P": np.asmatrix(P),
        "state": state,
        "score": float(score),
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    # vi: TODO Part H — chỉ lidar: hit +1/window (tối đa 1), miss trong FOV -1/window.
    # vi: score > confirmed_threshold → confirmed; đã confirmed không hạ trạng thái.
    # vi: Camera không gọi hàm này; track chưa confirmed với hit → tentative.
    params = get_tracking_params()
    delta = 1.0 / params.window

    score = track["score"]
    state = track["state"]

    if associated:
        score = min(1.0, score + delta)
        if state != "confirmed":
            if score > params.confirmed_threshold:
                state = "confirmed"
            else:
                state = "tentative"
    else:
        score = score - delta

    track["score"] = float(score)
    track["state"] = state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    # vi: TODO Part H — Pxx hoặc Pyy > max_P: xóa bất kể score.
    # vi: confirmed: xóa khi score < delete_threshold; chưa confirmed: score <= 0.
    # vi: Các điều kiện là OR; camera không đánh giá/xóa track.
    params = get_tracking_params()

    P = track["P"]
    score = track["score"]
    state = track["state"]

    # 1. Phương sai vị trí ngang (P[0, 0] hoặc P[1, 1]) vượt quá giới hạn
    if P[0, 0] > params.max_P or P[1, 1] > params.max_P:
        return True

    # 2. Xóa theo điểm số tồn tại (score)
    if state == "confirmed":
        if score < params.delete_threshold:
            return True
    else:
        if score <= 0:
            return True

    return False