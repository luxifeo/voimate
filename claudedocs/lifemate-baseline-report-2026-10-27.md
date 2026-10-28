# LifeMate baseline: Gemma-4-E2B, không huấn luyện lại (2026-10-27)

## 1. Cấu hình baseline cuối (v3)

| Thành phần | Lựa chọn |
|---|---|
| Core LLM + ASR | `google/gemma-4-E2B-it` (bản instruction-tuned của gemma-4-E2B), vLLM 0.32.0, 1×A30 |
| ASR | Gemma audio input, prompt ASR của model card + danh sách app đã cài làm ngữ cảnh |
| Agent | native tool calling (`--tool-call-parser gemma4`), **thinking bật**, temperature 0, tối đa 10 bước |
| Few-shot | `BAAI/bge-m3` embed transcript; top-4 mẫu train có app nằm trong `installed_apps` của mẫu |
| Chống lỗi schema | `env.validate()` trước `env.call()`. Call sai schema được trả lỗi cho model và không vào log nộp bài |
| Nudge | model trả lời bằng câu hỏi/hứa hẹn ("?", "sẽ") → nhắc 1 lần phải gọi tool hoặc `sys.ask_missing_info` |

## 2. Kết quả

Metric: tái hiện `evaluation.md` trong `lifemate/scorer.py`. Kiểm chứng: replay gold → 100.0. Baseline luôn trả `[]` → 7.18 trên full train.

### 2.1 Agent: dev 300 mẫu train (seed 0)

| Run | Đầu vào lượt cuối | Thinking | Prompt | Score | State acc |
|---|---|---|---|---|---|
| v1_gold | transcript gold | tắt | v1 | 40.03 | 56.33 |
| v1_asr | ASR (prompt gốc) | tắt | v1 | 14.46 | 33.00 |
| v1_e2e_audio | audio trực tiếp | tắt | v1 | 12.22 | 30.67 |
| v1_e2e_audiotext | audio + ASR | tắt | v1 | 13.49 | 32.67 |
| v1_gold_think | transcript gold | bật | v1 | 54.40 | 72.67 |
| v1_asr_think | ASR (prompt gốc) | bật | v1 | 17.10 | 34.67 |
| v2_gold_think_nonudge | transcript gold | bật | v2, không nudge | 54.30 | 74.67 |
| v2_asr_think_nonudge | ASR (prompt gốc) | bật | v2, không nudge | 14.72 | 33.67 |
| v2_gold_think | transcript gold | bật | v2 + nudge | 54.02 | 72.67 |
| v2_asr_think | ASR (prompt gốc) | bật | v2 + nudge | 16.97 | 35.00 |
| v2_asrctx_think | ASR (prompt ngữ cảnh) | bật | v2 + nudge | 17.56 | 38.67 |
| v3_gold_think | transcript gold | bật | v3 + nudge | 53.77 | 71.00 |
| **v3_asrctx_think** | ASR (prompt ngữ cảnh) | bật | v3 + nudge | **19.15** | 39.33 |

**Full train (1545 mẫu), v3 + ASR ngữ cảnh: 18.17**. Theo nhóm: action 14.46, ask_missing_info 21.06, out_of_scope 53.15. State acc 38.25. Tốc độ 0.81 s/mẫu với 48 luồng song song.

Với n=300, sai số chuẩn khoảng ±2.8 điểm. Thêm vào đó, vLLM batching không xác định nên chạy lại cùng cấu hình cũng lệch vài điểm. Vì vậy các chênh lệch < 3 điểm trong bảng chưa đủ để kết luận.

### 2.2 ASR: WER/CER lượt cuối

| Biến thể (300 clip, seed 1) | WER | CER |
|---|---|---|
| prompt model card (gốc) | 43.8% | 27.0% |
| prompt tiếng Việt | 43.2% | 26.2% |
| + ngữ cảnh app (**dùng**) | 42.0% | 26.8% |
| system prompt ngữ cảnh | 42.1% | 26.0% |
| audio đặt trước text | 47.7% | 29.9% |
| chuẩn hoá peak âm lượng | 44.3% | 27.1% |
| pass 2: audio + bản nháp + ngữ cảnh | 42.3% | 26.2% |
| sửa text bằng 6 cặp (ASR → đúng) retrieve từ train | 44.6% | 30.8% |
| n-best (1 greedy + 6 mẫu T=0.7) chọn medoid | 41.8% | 26.7% |

Full train: prompt gốc WER 45.2% / CER 27.8%; prompt ngữ cảnh WER 43.8% / CER 28.5%.

## 3. Nhận định

1. **ASR là nút thắt chính.** Cùng agent v3, transcript gold đạt 53.77, transcript ASR đạt 19.15. Tất cả cách cải thiện ASR chỉ dùng inference đều chỉ đưa WER về khoảng 42%.
2. **Thinking mode là cải tiến lớn nhất phía agent**: +14.4 điểm trên transcript gold (40.03 → 54.40), thời gian mỗi mẫu gần như không đổi.
3. **End-to-end audio không tốt hơn cascade** (12.2–13.5, so với 14.5 khi dùng transcript).
4. **Lạm dụng `ask_missing_info` khi ASR méo**: dev v3 ASR có 43.7% mẫu bị hỏi lại, trong đó 87 mẫu gold là hành động. Với transcript gold, tỉ lệ này là 24%.
5. Nudge giúp nhóm ask_missing_info (ASR: 11.0 → 15.7) nhưng tổng điểm chênh nằm trong mức nhiễu.

## 4. Lưu ý về độ tin cậy

- [Inference] Điểm train có thể cao hơn public/private. Few-shot lấy từ chính train (chỉ loại đúng mẫu đang chấm), trong khi train có nhiều câu cùng template. Ngoài ra public_test có 11 app mới không có trong train (parking, grocery, recipes, train_bus_tickets…), nên các app này không có ví dụ few-shot.
- [Unverified] Scorer local giả định đối số non-fuzzy phải bằng nhau tuyệt đối và tập key phải trùng. `evaluation.md` không nói rõ hai điểm này.
- Submission public_test `outputs/runs/public_test/v3/submission.jsonl`: 634 dòng, validator của BTC báo 0 lỗi schema. **Chưa có điểm public** vì cần nộp lên leaderboard.

## 5. Docker nộp bài (bản nháp)

`Dockerfile` + `docker/entrypoint.sh` (image `lifemate-submission:baseline`, 36.3GB, model nằm sẵn trong image, chạy offline). Đã test trên 30 mẫu public_test: 0 lỗi schema; vLLM khởi động ~4.5 phút. Hack CUDA forward-compat chỉ truyền qua `-e LD_LIBRARY_PATH` lúc `docker run` trên máy này (driver 535), không có trong Dockerfile. Mount `/data`, `/output` và biến `SPLIT` là giả định, cần chỉnh theo hướng dẫn đóng gói chính thức của BTC.

## 6. Hướng tiếp theo (xếp theo tác động kỳ vọng)

1. [Inference] **Model ASR tiếng Việt chuyên dụng** (vd. PhoWhisper, Whisper large-v3) thay cho ASR của Gemma, nếu quy định cuộc thi và bạn cho phép thêm model. Đây là hướng có tiềm năng lớn nhất, vì khoảng cách gold–ASR là ~35 điểm.
2. Fine-tune (LoRA) Gemma-4-E2B cho ASR và/hoặc tool calling trên 1545 mẫu train. Hướng này nằm ngoài phạm vi "không training" của baseline.
3. Hiệu chỉnh ngưỡng ask_missing_info khi transcript nhiễu (mục 3.4).
4. Đánh giá trên tập dev giữ lại theo app (holdout app) để mô phỏng app mới của public/private.
