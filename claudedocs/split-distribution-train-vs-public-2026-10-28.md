# So sánh phân bố train và public_test (2026-10-28)

Script: `scripts/compare_splits.py`. Số liệu đầy đủ: `outputs/analysis/split_shift.json`.

Dữ liệu dùng: input của `<split>.jsonl`, `<split>_tools_catalog.json`, audio stats (`outputs/asr/audio_stats.json`), transcript ASR của chính pipeline (`outputs/asr/<split>_ctx.json`), dự đoán baseline v3 trên cả hai split. Không dùng nhãn public_test (không có). Không đọc `<split>_environment/`.

Test thống kê: KS cho biến số, chi-square + Jensen-Shannon divergence (JSD, 0 = giống hệt, 1 = tách rời) cho biến phân loại. Adversarial validation: logistic regression phân biệt train vs public_test, 5-fold, ROC-AUC ≈ 0.5 nghĩa là không phân biệt được.

## 1. Kết luận

Phân bố **không đồng nhất ở tập app/tool**. Các chiều còn lại (history, context, chất lượng audio) gần như giống nhau.

| Chiều | Kết quả | Mức lệch |
|---|---|---|
| Tập app trong catalog | 11 app chỉ có ở public_test (34 tool), 21 app chỉ có ở train | **Lớn** |
| App đã cài trên mỗi mẫu | 64.7% mẫu public_test có ≥1 app không có trong train | **Lớn** (AUC 0.972) |
| Tỷ lệ cài các app chung | JSD 0.0033 | Không đáng kể |
| Schema 88 tool chung | Giống hệt (cùng schema 2.7.0) | Không |
| History | Có history 65.9% vs 66.1%; số lượt KS p = 0.33 | Không |
| Context (city, timezone, profile, weekday) | chi2 p > 0.4 | Không |
| Giờ trong ngày, tháng | 9h: 31.8% vs 36.6%; chi2 p ≈ 0.03–0.04 | Nhỏ |
| Audio (SNR, noise, peak, clipping) | KS p > 0.5 | Không |
| Độ dài audio | 7.23s vs 7.70s, KS 0.073 (p 0.016) | Nhỏ |
| Nội dung transcript (bge-m3) | AUC 0.596; lân cận gần nhất trong train thấp hơn | Vừa |

Adversarial AUC theo nhóm đặc trưng:

| Đặc trưng | AUC |
|---|---|
| Tabular, có toàn bộ app | 0.972 ± 0.007 |
| Tabular, chỉ app chung | 0.650 ± 0.016 |
| Context + audio, không có app | 0.579 ± 0.034 |
| Embedding transcript (bge-m3) | 0.596 ± 0.030 |

## 2. Chi tiết

### 2.1 Catalog

- App chỉ có ở public_test: `ecommerce`, `event_tickets`, `grocery`, `habit_tracker`, `pantry`, `parking`, `public_transit`, `recipes`, `stocks`, `train_bus_tickets`, `translator`.
- App chỉ có ở train: `appliances`, `bike_rental`, `calculator`, `door_lock`, `exchange_rate`, `fitness`, `home_automation`, `home_energy`, `home_security`, `household_chores`, `learning`, `library`, `lottery`, `loyalty`, `medication`, `news`, `package_tracking`, `podcast_radio`, `robot_vacuum`, `subscriptions`, `video_meeting`.
- Tỷ lệ cài app mới trên public_test: grocery 16.9%, recipes 13.9%, parking 13.7%, public_transit 12.9%, pantry 11.7%, train_bus_tickets 11.7%, ecommerce 10.6%, event_tickets 10.1%, stocks 9.3%, translator 7.9%, habit_tracker ~6%.
- Nhóm app: `food` 3.2% → 8.5%, `mobility` 6.4% → 10.0%, `travel` 0.7% → 2.7%; `home` 10.6% → 4.1%, `health` 3.2% → 0.6%.

### 2.2 Nội dung yêu cầu

- Transcript public_test dài hơn một chút: 22.2 → 23.7 từ (KS p 0.015).
- 4.6% token và 31.2% từ vựng của public_test không có trong train.
- Từ xuất hiện nhiều hơn hẳn ở public_test: `tủ`, `buýt`, `bãi`, `nấu`, `tàu`, `cổ` (phiếu), `vé`, `metro`, `chuyến`, `dịch`, `món`. Các từ này khớp với domain của app mới.
- 25 transcript "giống public_test nhất" theo classifier đều thuộc nấu ăn / tủ bếp / đi chợ / vé sự kiện / vé tàu xe / trả hàng.
- Cosine với mẫu train gần nhất (bge-m3): trung vị 0.72 (train, leave-one-out) vs 0.70 (public_test); tỷ lệ < 0.7: 38.2% vs 49.1%.

### 2.3 Dự đoán baseline v3 (proxy, không phải nhãn)

- Phân bố loại output (empty / ask_missing_info / action) giống nhau giữa hai split (chi2 p 0.22).
- Lưu ý: trên train, baseline dự đoán ask_missing_info 43.7% trong khi gold chỉ 14.0%. Vì vậy loại output dự đoán **không** phản ánh đúng phân bố thật.
- Baseline gọi app mới trong 15.0% mẫu public_test (event_tickets 20, recipes 15, ecommerce 10, ...).

### 2.4 Lỗi phụ phát hiện được

Transcript ASR lặp lại danh sách tên app từ prompt ngữ cảnh (ví dụ `public_0433`: "Công thức nấu ăn, Xe buýt/Metro, Nhà thông minh, ..."). Tỷ lệ: train 2.3% (35 mẫu), public_test 2.1% (13 mẫu). Tỷ lệ bằng nhau nên đây không phải shift, mà là lỗi của prompt ASR.

## 3. Ảnh hưởng

[Inference] Các điểm dưới đây suy ra từ số liệu trên, chưa được kiểm chứng bằng điểm leaderboard.

1. Điểm local trên train có thể cao hơn điểm public. Few-shot retriever chỉ chọn mẫu train có app nằm trong `installed_apps`. Với request dùng app mới, không có ví dụ nào, agent chạy zero-shot.
2. Mỗi split có catalog và `zaic_tools` riêng. [Speculation] private_test có thể lại có app mà cả train lẫn public_test đều không có. Khả năng xử lý tool chưa thấy quan trọng hơn khớp với tập app của public_test.
3. Dev split ngẫu nhiên từ train không đo được khả năng này, vì mọi app trong dev đều đã có trong phần train còn lại.

## 4. Đề xuất

1. **Dev split giữ lại app (leave-app-out):** chọn ~10 app trong train, đưa mọi mẫu có gold call tới các app đó vào dev, và loại chúng khỏi pool few-shot. Điểm trên dev này mô phỏng tình huống app chưa thấy.
2. **Few-shot theo cấu trúc thay vì theo app:** khi không có ví dụ cùng app, lấy ví dụ cùng mẫu hành vi (search → write, read → điều kiện → write, ask_missing_info) từ app khác.
3. **Sửa prompt ASR** để không lặp danh sách app (2.2% mẫu).
4. Dữ liệu synthetic (LLM + TTS) được phép trong lúc phát triển. Nên sinh cho nhiều app giả định khác nhau, không chỉ 11 app của public_test, để tránh khớp quá mức với public.
