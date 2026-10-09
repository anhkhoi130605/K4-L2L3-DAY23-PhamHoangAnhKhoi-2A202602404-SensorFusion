# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Phạm Hoàng Anh Khôi
- MSSV: 2A202602404
- Email: 
- Link repo (fork): https://github.com/anhkhoi130605/K4-L2L3-DAY23-PhamHoangAnhKhoi-2A202602404-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): 327ea36eda0a086c8d3b5bbae951945176b0f106

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, [0, 198], training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9951515151515151, 0.5757363253856943, 821/4/605
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.21534260752765794, 746, 34.59383920814095, 3, 680, 3.763819095477387
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.2207396543230981, 746, 36.34959226304792, 3, 680, 3.763819095477387
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: lidar cho RMSE thấp hơn fused (0.2153 so với 0.2207) dù cả hai có cùng số matches 746 và cùng ghost/miss (3/680), nên camera trong cấu hình hiện tại chưa cải thiện quỹ đạo track mà còn làm tăng sai số định vị. Điều này hợp lý vì `fusion-run-lab --fusion compare` đang đánh giá track-then-fuse: detection/track core vẫn dựa trên lidar, camera chỉ đi vào EKF update nên tác động nhỏ hơn lidar ở các frame có đo ổn định.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường: `mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`, `sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và `matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`. File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`, `grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)? Lidar cập nhật với đo 3D của object trong hệ BEV/ego nên `z` thường có đủ tọa độ không gian và covariance 3x3; camera chỉ cho thông tin hình học chiếu từ ảnh, thường là đo 2D/ít chiều hơn và `R` phải phản ánh sai số pixel + bất định chiếu, nên innovation khác scale và nhạy với calibration hơn.
2. Vì sao cần gating Mahalanobis trước khi gán? Vì gating loại các cặp track-measurement có innovation quá lớn so với covariance dự đoán, tránh gán nhầm ở các object gần nhau, giảm FP association và giữ cập nhật ổn định.
3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`. Đây là track-then-fuse: pipeline vẫn tạo/duy trì track bằng lidar trước, sau đó mới dùng camera để hiệu chỉnh state trong EKF; điều này thấy qua việc tracker core và score/init/delete vẫn bám vào lidar, còn camera chỉ xuất hiện như một sensor update bổ sung trong log compare.
4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual? Residual camera sẽ có bias có hướng, tăng theo khoảng cách/ góc nhìn, các innovation theo trục image hoặc trục tương ứng sẽ không còn zero-mean; đồng thời Mahalanobis distance tăng ở các frame xa/cạnh ảnh, dễ làm update bị reject hoặc kéo track lệch dần.
5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Vì cùng một frame rỗng vẫn phải phân biệt sensor nào đang cập nhật để áp đúng measurement model, covariance, và logic score/init/delete; lidar dùng để quyết định score/init/delete, còn camera chỉ hỗ trợ EKF update nên nếu không truyền sensor thì frame rỗng không biết áp nhánh nào.
6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track. Track thường được xác nhận khi đủ số lần match liên tiếp/tích lũy theo ngưỡng `n_init`; sau đó track confirmed vẫn được giữ qua vài frame miss nhờ bộ đếm `time_since_update` và ngưỡng `max_age`; xóa track khi miss vượt ngưỡng sống hoặc chưa bao giờ đủ điều kiện xác nhận mà bị mất trước khi chắc chắn.

## Bonus (không bắt buộc)

Không

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): GitHub Copilot CLI
- Dùng cho phần nào (hàm, câu hỏi, debug): điền báo cáo nộp, tổng hợp metrics/log cuối cùng, diễn đạt phần giải thích 6 câu và kiểm tra các trường cần nêu trong submission
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): đối chiếu trực tiếp `student/artifacts/metrics.json` và `student/artifacts/grade_run*.log`, rồi chạy `python tools/check_submission.py` sau khi cập nhật báo cáo

## Checklist nộp

- [ ] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [ ] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [ ] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [ ] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
