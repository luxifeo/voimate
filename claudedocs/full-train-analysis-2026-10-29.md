# Run toàn bộ train, cấu hình c1_combo, transcript ASR (2026-10-29)

Run: `outputs/runs/train/full_combo_asr/`.

Lệnh chạy:

```bash
uv run python scripts/run_agent.py --split train --transcripts asr --asr-cache _gip15 --model qwen3.5-4b --thinking \
  --max-tokens 4096 --fewshot 8 --arg-guide --date-table weeks --notes-folds 5 --name full_combo_asr
```

`--notes-folds 5`: ghi chú tham số của mỗi mẫu học từ 4 fold còn lại. Few-shot loại chính mẫu đang chạy.

| File | Nội dung |
|---|---|
| `llm_calls.jsonl` | 151 MB, mọi request/response LLM; xem bằng `lab/app.py` (cổng 8765) |
| `traces.jsonl` | trace agent từng mẫu |
| `scores.jsonl` | điểm từng mẫu |
| `metrics.json` | điểm tổng và cấu hình |
| `error_analysis.jsonl` | loại lỗi từng mẫu chưa đạt điểm tối đa (`scripts/error_analysis.py full_combo_asr`) |

## Điểm

| Nhóm | n | Score | State acc | Precision | Recall |
|---|---|---|---|---|---|
| Tất cả | 1,545 | **50.94** | 67.77 | 60.98 | 62.66 |
| action | 1,217 | 51.90 | 67.21 | 62.75 | 64.75 |
| ask_missing_info | 217 | 42.54 | 59.91 | 53.25 | 53.96 |
| out_of_scope | 111 | 56.76 | 89.19 | 56.76 | 56.76 |

- Tốc độ: 3.93 s/sample.
- 581 mẫu đạt điểm tối đa.
- Dev 300 của cùng cấu hình (`c1_combo_asr`) đạt 48.84. Public LB đạt 47.39.

## Lỗi hạ tầng: 18 mẫu, điểm trung bình 0.06

| Lỗi | Số mẫu |
|---|---|
| `APITimeoutError` (client timeout 300 s, KV cache vLLM đầy khoảng 98%) | 15 |
| Prompt vượt 16,384 token (8 few-shot + history dài) | 3 |

Đường nộp bài (`predict.py`) dùng cùng timeout và cùng cấu hình few-shot, nên cũng có nguy cơ gặp hai lỗi này. Run public trước đó có 0 lỗi.

## Điểm mất theo loại lỗi (tổng mất 49.06 điểm)

| Loại | Mẫu | Điểm mất | % mất |
|---|---|---|---|
| Write call đúng tool, sai đối số | 279 | 15.08 | 31% |
| Chỉ lệch read call | 253 | 6.85 | 14% |
| Thiếu write call | 104 | 6.68 | 14% |
| Hỏi thừa (gold hành động) | 96 | 6.02 | 12% |
| Thiếu hỏi (gold ask_missing_info) | 80 | 5.06 | 10% |
| Write call thừa | 62 | 3.62 | 7% |
| OOS nhưng vẫn gọi tool | 48 | 3.11 | 6% |
| Ask sai nội dung | 21 | 1.29 | 3% |
| Không gọi tool nào | 20 | 1.29 | 3% |

## Mẫu hình đáng chú ý

### Read call trong gold hầu như không có đối số

Model thì thêm bộ lọc. Phần lớn lỗi `extra_key` đến từ đây (`calendar.list_events.date_from/date_to` 41+41, `smart_home.get_status.floor` 24, `bill_payment.list_bills.type` 23).

| Tool | Gold: call không đối số | Model: call không đối số |
|---|---|---|
| `calendar.list_events` | 83% | 40% |
| `smart_home.get_status` | 87% | 56% |
| `bill_payment.list_bills` | 100% | 79% |

Ngưỡng hiện tại của `param_usage_notes` (<10% mới ghi "hầu như không dùng") bỏ sót các tham số được điền 10–20%.

### `contacts.search.query`

- Có 114 lỗi giá trị, nhiều nhất trong các đối số.
- Trong gold, query xuất hiện nguyên văn trong câu yêu cầu ở 197/204 call.
- Model hay bỏ danh xưng ("Sếp Dũng" → "Dũng"). Với transcript ASR, lỗi nhận dạng tên riêng cũng rơi vào đây.

### App điểm thấp nhất (≥ 8 mẫu)

loyalty 15.8, email 25.2, lottery 26.8, fuel_price 27.4, maps 28.3, bike_rental 29.8.
