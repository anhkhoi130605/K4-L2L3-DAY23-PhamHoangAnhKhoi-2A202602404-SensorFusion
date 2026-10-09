"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params

# vi: from fusion_lab.workspace_support import get_tracking_params
# vi: from fusion_lab.workspace_loader import load_workspace_module
# vi: kalman = load_workspace_module("kalman")  # không dùng `import kalman`
kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    # vi: TODO Part F — H = meas.sensor.get_H(track.x);
    # vi: gamma = kalman.innovation(...); S = kalman.innovation_covariance(...);
    # vi: return gamma.T @ inv(S) @ gamma (float scalar).
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    d2 = gamma.T @ np.linalg.inv(S) @ gamma
    return float(np.asarray(d2).item())


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    # vi: TODO Part F — ngưỡng chi2.ppf(gating_threshold, sensor.dim_meas) từ params.
    params = get_tracking_params()
    gate_limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
    return bool(mhd_sq <= gate_limit)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    # vi: TODO Part F — khởi tạo toàn inf; kiểm tra meas.sensor.in_fov(track.x)
    # vi: trước MHD (camera sau lưng/độ sâu 0 không được chiếu); rồi kiểm tra chi2.
    num_tracks = len(track_list)
    num_meas = len(meas_list)
    cost = np.full((num_tracks, num_meas), np.inf)

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            if meas.sensor.in_fov(track.x):
                mhd_sq = mahalanobis_distance(track, meas)
                if chi2_gate(mhd_sq, meas.sensor):
                    cost[i, j] = mhd_sq

    return np.asmatrix(cost)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    # vi: TODO Part F — chỉ lấy cặp hữu hạn nhỏ nhất rồi xóa hàng/cột tương ứng;
    # vi: ma trận rỗng/toàn inf: trả np.nan, np.nan và giữ các danh sách chưa ghép.
    mat = np.asarray(association_matrix)

    if mat.size == 0 or np.all(np.isinf(mat)):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)

    min_idx = np.unravel_index(np.argmin(mat), mat.shape)
    r, c = int(min_idx[0]), int(min_idx[1])

    if np.isinf(mat[r, c]):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)

    track = unassigned_tracks[r]
    meas = unassigned_meas[c]

    remaining_tracks = [t for idx, t in enumerate(unassigned_tracks) if idx != r]
    remaining_meas = [m for idx, m in enumerate(unassigned_meas) if idx != c]

    new_matrix = np.delete(mat, r, axis=0)
    new_matrix = np.delete(new_matrix, c, axis=1)

    return track, meas, np.asmatrix(new_matrix), remaining_tracks, remaining_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    # vi: TODO Part F — kể cả meas_list rỗng, vẫn gọi quản lý cuối lượt.
    # vi: Ghép cặp hữu hạn, filter_obj.update rồi handle_updated_track(track, sensor).
    # vi: Không bỏ qua FOV sau khi đã xóa cặp khỏi danh sách chưa ghép.
    # vi: Kết thúc manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor).
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    if len(unassigned_tracks) > 0 and len(unassigned_meas) > 0:
        cost_mat = association_cost_matrix(unassigned_tracks, unassigned_meas)

        while True:
            track, meas, cost_mat, unassigned_tracks, unassigned_meas = pick_next_pair(
                cost_mat, unassigned_tracks, unassigned_meas
            )
            if track is np.nan or meas is np.nan:
                break

            filter_obj.update(track, meas)
            manager.handle_updated_track(track, sensor)

    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)