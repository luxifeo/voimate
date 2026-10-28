# Kiểm tra 7 giả thiết cải thiện agent (2026-10-28)

Mọi run: Qwen3.5-4B trên vLLM, thinking, max_tokens 4096, dev 300 (`random.Random(0).sample(train, 300)`).
Giả thiết 1–6 chạy trên transcript gold, để tách lỗi LLM khỏi lỗi ASR. Cấu hình tốt nhất được chạy lại trên transcript ASR gipformer (`_gip15`).
So sánh: `uv run python scripts/compare_runs.py <base> <run>...`. Kiểm tra lỗi ngày: `scripts/reasoning_date_audit.py <run>`.

## Nhiễu đo lường

vLLM ở temperature 0 không batch-invariant. Ba run baseline giống hệt cấu hình cho 63.26 / 60.72 / 62.78 điểm (trung bình 62.25).
Chênh lệch theo cặp giữa hai run cùng cấu hình có SE khoảng 1.1–1.2 điểm. Vì vậy một run đơn lẻ chênh dưới khoảng 2.5 điểm chưa đủ làm bằng chứng.
Do đó mỗi giả thiết được đánh giá thêm bằng chỉ số riêng cho nó: số đối số thừa, số câu gán thứ cho ngày bị sai trong reasoning, số giá trị ngày giờ sai, số token.

## Kết quả (gold, dev 300; Δ là chênh theo cặp so với `h0_base_gold`)

| Giả thiết | Run | Điểm | Δ ± SE | ↑/↓ mẫu | Đối số thừa | Giá trị ngày giờ sai | Prompt tok | s/mẫu | Kết luận |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | h0_base_gold | 62.78 | — | — | 101 | 8 | 7455 | 2.89 | |
| 1. Lọc tool (oracle: chỉ tool của app gold) | h1_oracle_gold | 60.85 | −1.93 ± 1.90 | 34/43 | 142 | 13 | 4386 | 1.89 | Không ủng hộ: ngay cả lọc hoàn hảo cũng không tăng điểm |
| 1. Lọc tool (thực tế: app của 16 ví dụ gần nhất + top-3 app theo mô tả + app trong history; giữ đủ app gold ở 97% mẫu) | h1_retr_gold | 64.07 | +1.29 ± 1.82 | 31/31 | 114 | 8 | 5841 | 2.32 | Trung tính về điểm, nhanh hơn ~20% |
| 2. Hướng dẫn điền tham số mặc định | h2_argguide_gold | 64.77 | +1.99 ± 1.75 | 49/31 | **66** | 9 | 7774 | 3.22 | **Ủng hộ** |
| 2. (chạy lặp lại) | h2_argguide_gold_r2 | 65.60 | +2.83 ± 1.69 | 47/24 | **66** | 11 | 7729 | 2.99 | |
| 3. Tóm tắt history (nén theo luật) | h3_summary_gold | 60.69 | −2.09 ± 1.79 | 29/43 | 119 | 10 | 7475 | 3.25 | Không ủng hộ |
| 3. Tóm tắt history (bằng LLM) | h3_llmsum_gold | 57.70 | −5.08 ± 2.10 | 32/57 | 112 | 16 | 5973 | 3.39 | Bị bác: giảm rõ |
| 4. Tool calculator | h4_calc_gold | 60.62 | −2.15 ± 1.62 | 23/27 | 109 | 13 | 7740 | 3.08 | Không ủng hộ: model chỉ gọi ở 15/300 mẫu |
| 5. Time context ở user message | h5_user_gold | 61.67 | −1.11 ± 2.22 | 41/45 | 113 | 9 | 7520 | 3.17 | Không giảm lỗi ngày (16 → 15 mẫu) |
| 5. Time context ở cuối system prompt | h5_end_gold | 61.39 | −1.39 ± 1.70 | 26/36 | 108 | 12 | 7488 | 3.21 | Không ủng hộ |
| (thêm) Bảng ngày có nhãn tuần, phủ từ thứ Hai tuần trước đến hết tuần sau nữa | h7_weeks_gold | 63.61 | +0.83 ± 1.76 | 31/25 | 108 | **6** | 7787 | 2.75 | Giảm lỗi ngày: mẫu lỗi 16 → 10; câu gán thứ sai 13/97 → 4/76 |
| 6. Không dùng few-shot | h6_fs0_gold | 47.24 | −15.54 ± 2.26 | 22/87 | 166 | 10 | 6381 | 2.64 | Bị bác |
| 6. Few-shot chỉ giữ lời gọi (bỏ kết quả và lượt trước) | h6_calls_gold | 57.21 | −5.57 ± 2.17 | 31/58 | 128 | 10 | 6700 | 2.57 | Bị bác: bớt thông tin là hại |
| 6. Ngược lại: 8 ví dụ | h6_fs8_gold | 65.71 | +2.93 ± 1.98 | 45/28 | 90 | 12 | 8546 | 3.05 | Nhiều ví dụ hơn thì tốt hơn |
| 6. Ngược lại: 12 ví dụ | h6_fs12_gold | 64.57 | +1.79 ± 1.89 | 45/32 | 90 | 12 | 9610 | 3.66 | Không hơn 8 ví dụ |
| **Kết hợp: 2 + 8 ví dụ + bảng ngày có nhãn tuần** | c1_combo_gold | **67.69** | **+4.91 ± 2.06** | 55/31 | 65 | 10 | 9162 | 3.41 | So với trung bình 3 baseline: **+5.44, CI 95% [+1.79, +9.02]** |

s/mẫu = tổng thời gian run / 300 mẫu, với 32 mẫu chạy song song trên 1 A30.

## Chi tiết từng giả thiết

**1. Nhiều tool quá.** Trung bình mỗi mẫu có 32.5 tool, trong khi gold chỉ dùng 1.5 app. Bản oracle, chỉ đưa tool của app gold (giảm 41% token prompt), vẫn không tăng điểm. Số đối số thừa lại tăng (101 → 142).
- Bộ lọc thực tế làm mất app cần dùng ở 9/300 mẫu. Điểm trên 9 mẫu đó giảm từ 34.3 xuống 18.9.
- [Inference] Model không bị rối vì danh sách tool. Lợi ích duy nhất là tốc độ.

**2. Thiếu hướng dẫn tham số mặc định.** Thống kê trên train (ngoài dev) cho thấy 47 tham số tùy chọn gần như không bao giờ được gold điền. Ví dụ:
- `reminders.create.repeat` (0/115 lần), `alarm.create.repeat` (0/78), `bill_payment.list_bills.type` (0/112), `banking.transfer.note` (1/82).

Ngược lại, một số tham số tùy chọn gold gần như luôn điền:
- `smart_home.light_set.power` (93%, gần như luôn là `"on"`), `calendar.update_event.start/end` (100%), `notes.create.title` (94%).

Cờ `--arg-guide` gắn ghi chú này vào mô tả từng tham số và thêm luật 12. Kết quả:
- **Đối số thừa giảm từ 101 xuống 66 ở cả 2 run.**
- Nhóm action tăng từ 63.6 lên khoảng 67.8.
- Trên ASR: xem bảng ở mục "Trên transcript ASR".

**3. Tóm tắt history.** Bản nén theo luật gần như không giảm token (7,455 → 7,475), vì kết quả tool vẫn chiếm phần lớn. Bản tóm tắt bằng LLM giảm còn 5,973 token. Cả hai đều làm giảm điểm trên 199 mẫu có history:
- nén theo luật: 62.06 → 59.33;
- tóm tắt bằng LLM: 62.06 → 55.66.

[Inference] Model cần nguyên văn kết quả tool và ID từ các lượt trước.

**4. Calculator.** Tool `scratch__calc` chạy cục bộ và không bao giờ gửi tới ToolEnv, nên không xuất hiện trong submission. Nó hỗ trợ số học, so sánh, cộng/trừ phút và ngày, và tra thứ của một ngày.
- Dù có luật bắt buộc, model chỉ gọi 16 lần trên 15/300 mẫu.
- [Inference] Các lỗi so sánh trong audit là do model không nhận ra cần tính, nên có thêm công cụ cũng không đổi.

**5. Vị trí time context.** Cả hai vị trí (user message và cuối system prompt) đều chỉ đổi chỗ lỗi:
- 12 mẫu được sửa, 11 mẫu lỗi mới.
- Lỗi mới chủ yếu lệch đúng 7 ngày với "thứ X tuần sau". Ví dụ train_1568: hôm nay thứ Sáu 13/11, "thứ ba tuần sau" là 17/11 nhưng model chọn 24/11.

Bảng ngày có nhãn tuần nhắm thẳng vào lỗi này và giảm lỗi ngày rõ hơn. Lỗi ngày chỉ chiếm khoảng 1–2 điểm, nên điểm tổng thay đổi ít.

**6. Ví dụ few-shot quá nhiều thông tin.** Kết quả ngược với giả thiết:

| Few-shot | Điểm |
|---|---|
| 0 ví dụ | 47.2 |
| 4 ví dụ, chỉ có lời gọi | 57.2 |
| 4 ví dụ đầy đủ (trung bình 3 run) | 62.25 |
| 8 ví dụ | 65.7 |
| 12 ví dụ | 64.6 |

Nhóm cần hỏi lại phụ thuộc mạnh vào ví dụ: 20.5 điểm khi không có ví dụ, 51.0 với 4 ví dụ, 55.4 với 8 ví dụ.

**7. Khử nhiễu / tách giọng trước ASR.** Xem [asr_enhancement.md](asr_enhancement.md).
- GTCRN, DPDFNet và htdemucs không giảm WER. Áp thẳng thì WER tăng 0.25 đến 12.75 điểm.
- Gate "htdemucs khi SNR < 15 dB" chỉ giảm 0.05 điểm WER trên 5,036 audio.
- WER ở clip sạch (≥ 40 dB) vẫn là 17%, nên phần lớn lỗi ASR không do nhiễu.
- Khuyến nghị: không thêm vào pipeline.

## Trên transcript ASR (gipformer `_gip15`, dev 300)

| Run | Điểm | Δ so với trung bình 2 baseline ASR | Đối số thừa | ask | oos |
|---|---|---|---|---|---|
| llm_qwen35_4b_asr_log (baseline) | 41.19 | | 106 | 26.5 | 62.5 |
| llm_qwen35_4b_asr (baseline) | 41.73 | | 108 | 28.5 | 58.3 |
| h2_argguide_asr | 44.21 | +2.75 ± 1.42 | 68 | 34.1 | 70.8 |
| **c1_combo_asr** (2 + 8 ví dụ + bảng ngày có nhãn tuần) | **48.84** | **+7.38 ± 1.88, CI 95% [+3.71, +11.00]** | 61 | 37.7 | 58.3 |

c1_combo_asr chạy 4.03 s/mẫu (wall/300 khi chạy 32 mẫu song song), prompt trung bình 9,133 token, trong giới hạn context 16K.

## Thay đổi code

Mọi cờ đều tắt mặc định. Với cấu hình mặc định, request gửi LLM giống hệt từng byte so với code gốc; đã kiểm tra trên 300 mẫu bằng cách so request JSON. `scripts/predict.py` và Dockerfile không bị sửa.

| File | Thay đổi |
|---|---|
| `lifemate/agent.py` | `AgentConfig`: `date_table`, `time_position`, `arg_guide`, `history_mode` (`summary` / `llm_summary`), `calculator`. `run_agent(apps=, param_notes=)`. Tool `scratch__calc` chạy cục bộ. |
| `lifemate/pipeline.py` | `build_tool_filters` (none / oracle / retrieval), `param_usage_notes` (ghi chú tham số từ nhãn train), tham số mới của `run_split` |
| `lifemate/retriever.py` | `format_example(style="calls")` |
| `scripts/run_agent.py` | cờ `--arg-guide --fewshot-style --tool-filter --time-position --date-table --history-mode --calculator` |
| `scripts/compare_runs.py` | mới: so sánh nhiều run, Δ theo cặp, chỉ số theo giả thiết |

`param_usage_notes` trong thí nghiệm được tính từ 1,245 mẫu train ngoài dev 300. Khi nộp bài thì tính từ toàn bộ train.

## Đề xuất

1. Bật `arg_guide` + `fewshot 8` + `date_table weeks` trong đường nộp bài. Đã áp dụng ngày 2026-10-29, xem mục Submission bên dưới.
2. Không lọc tool, không tóm tắt history, không dùng calculator, không khử nhiễu ASR. Nếu cần tốc độ thì lọc tool thực tế là lựa chọn duy nhất không làm giảm điểm.
3. Hai nhóm lỗi lớn còn lại (gold, cấu hình kết hợp):
   - write call sai đối số: 10.8 điểm mất;
   - read call lệch: 6.7 điểm mất.
4. Chênh lệch gold và ASR vẫn khoảng 20 điểm. Phần lớn do lỗi tên riêng và từ tiếng Anh trong ASR, không do nhiễu.

## Submission (2026-10-29, branch `feature/combo-submission`)

Cấu hình: gipformer 1.5 ASR + Qwen3.5-4B, `--fewshot 8 --arg-guide --date-table weeks`, `max_tokens` 4096, thinking. `param_usage_notes` tính từ toàn bộ 1,545 mẫu train.

### File nộp public_test

- Đường dẫn: `outputs/runs/public_test/c1_combo_gip15/submission.jsonl`, sinh bằng `scripts/run_agent.py` trên server vLLM dev.
- Kết quả chạy: 634 dòng, 0 lỗi schema (`zaic_tools --validate`), 0 lỗi agent, 4.43 s/sample.
- So với run public trước (`qwen35_4b_gip15`): cùng state_hash 476/634, cùng tool call 253/634.

### Docker `lifemate-submission:combo`

| File | Thay đổi |
|---|---|
| `scripts/predict.py` | ASR gipformer, 8 few-shot, `param_usage_notes`, `AgentConfig(date_table="weeks", arg_guide=True, max_tokens=4096)` |
| `docker/entrypoint.sh` | vLLM phục vụ Qwen3.5-4B: `--max-model-len 16384`, parser `qwen3_coder`/`qwen3`, `--seed 0` |
| `docker/fetch_models.py` | tải weights pin revision, ghi `refs/main` để chạy offline |
| `Dockerfile` | thêm sherpa-onnx 2.0.1, scipy, scikit-learn; bỏ thư viện audio của Gemma |

Entrypoint cũ đặt `--max-model-len 32768`, vượt giới hạn context 16,384 của rule.md. Bản mới đặt 16384.

Số tham số (đếm từ checkpoint):

| Model | Tham số |
|---|---|
| Qwen3.5-4B (LM 4.21B + vision 0.33B + MTP 0.12B) | 4.66B |
| bge-m3 | 0.57B |
| gipformer | 0.07B |
| **Tổng** | **khoảng 5.3B** |

### Smoke test Docker

Cấu hình: `--network none`, `--limit 40`, GPU0, `GPU_UTIL=0.75`.

| Hạng mục | Kết quả |
|---|---|
| Lỗi | 0 lỗi agent |
| Thời gian | 226 s pipeline; tổng 445 s gồm cả khởi động vLLM |
| ASR | trùng 95/95 transcript với `outputs/asr/public_test_gip15.json` |
| Agent so với file nộp ở trên | cùng state_hash 34/40, cùng tool call 27/40 |

Các mẫu lệch khác nhau ở quyết định của model (chọn chuyến bay, hỏi lại hay thực hiện), không lệch ngày giờ.

Trên máy dev, GPU0 có sẵn một `tritonserver` khác chiếm 3.5 GB. Vì vậy mức mặc định `GPU_UTIL=0.85` không khởi động được, phải truyền `-e GPU_UTIL=0.75` lúc chạy.
