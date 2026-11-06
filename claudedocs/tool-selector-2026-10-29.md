# Sub-agent chọn app và tool (2026-10-29)

## Thiết lập

| Thành phần | Mô tả |
|---|---|
| Code | `lifemate/tool_selector.py` (selector), `scripts/eval_tool_selector.py` (đánh giá) |
| Model | Qwen3.5-4B trên vLLM đang host (`qwen3.5-4b`, cổng 8011), temperature 0 |
| Gọi LLM | 1 call cho mỗi mẫu |
| Input | rule chọn tool; app đã cài kèm mô tả; mỗi tool có loại read/write, mô tả và tên tham số (`*` = bắt buộc); history (lời người dùng, tool call kèm kết quả cắt 300 ký tự, lời trợ lý); câu yêu cầu cuối |
| Output | JSON `{"apps": [...], "tools": [...]}`. vLLM ép theo JSON schema, chỉ cho phép tên trong danh sách app/tool đã cài |
| Few-shot | 8 mẫu train gần nhất theo bge-m3 (loại chính mẫu). Mỗi ví dụ gồm lượt người dùng trước, câu yêu cầu và tập tool gold |
| Gold | tool của `label.tool_calls` trừ `sys.ask_missing_info`, cộng `tool_name` mà ask hỏi tới. Mẫu out-of-scope có tập rỗng |
| Tập đo | dev 300 (seed 0), transcript ASR gipformer `_gip15`. Trung bình 32.4 tool đã cài và 2.11 tool gold mỗi mẫu |
| Log | `outputs/tool_selector/train/<run>/`: `llm_calls.jsonl` (mọi request/response), `predictions.jsonl`, `metrics.json` |

## Kết quả

Cột "bao phủ" là tỉ lệ mẫu có đủ mọi tool gold trong tập chọn. Đây là chỉ số quan trọng nếu dùng selector để lọc tool cho agent.

| Selector | Tool P | Tool R | Bao phủ tool | Đúng y tập tool | Bao phủ app | Số tool chọn | Tool thuộc app chọn | OOS → rỗng | s/mẫu |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot, không thinking | 68.2 | 62.6 | 46.0% | 30.7% | 72.0% | 1.94 | 5.48 | 45.8% | 0.15 |
| zero-shot, thinking | 77.0 | 77.9 | 70.0% | 51.3% | 80.7% | 2.14 | 5.38 | 62.5% | 0.54 |
| 8-shot, không thinking | 77.1 | 77.4 | 63.0% | 45.0% | 85.7% | 2.12 | 5.85 | 41.7% | 0.41 |
| **8-shot, thinking** | **80.2** | **86.9** | **80.0%** | **58.0%** | **89.0%** | 2.29 | 5.79 | 66.7% | 0.83 |
| Bộ lọc retrieval hiện tại (bge-m3, k=16 + top-3 mô tả) | 10.6 | 98.1 | 96.7% | 0% | 96.7% | 19.63 | 19.63 | 0% | 0.35 |

Bao phủ tool theo category gold:

| Selector | action (227) | ask_missing_info (49) |
|---|---|---|
| zero-shot, thinking | 67.8% | 65.3% |
| 8-shot, thinking | 80.2% | 69.4% |
| retrieval | 96.5% | 95.9% |

Mẫu out-of-scope có gold rỗng nên luôn được tính là bao phủ (100%).

## Phân tích (8-shot, thinking)

- **Tool bỏ sót:** 50 read, 33 write. Tool chọn thừa: 92 read, 44 write. Read tool vẫn là phần mơ hồ nhất (`contacts.search` thừa 15 lần, thiếu 4 lần).
- **Mẫu chưa bao phủ:** 60 mẫu, trong đó 27 mẫu chỉ thiếu read tool.
- **Tính riêng write tool và tool cần hỏi:** bao phủ 89.0%. Nếu mở rộng ra mọi tool của app được chọn thì đạt 93.7%.
- **Gộp app của nhiều biến thể** (agent thấy mọi tool của các app đó):

  | App gộp từ | Bao phủ | Số tool trung bình |
  |---|---|---|
  | 8-shot thinking | 89.0% | 5.8 |
  | + zero-shot thinking | 91.7% | 6.3 |
  | + 8-shot không thinking | 94.0% | 6.7 |

## Kết luận

- Thinking tăng bao phủ tool từ 46% lên 70% (zero-shot), và 8-shot tăng thêm lên 80%.
- [Inference] Trên public, 41% mẫu dùng app không có trong train, nên few-shot không có ví dụ đúng app. Độ chính xác trên các mẫu đó có thể gần mức zero-shot (70%) hơn là mức 8-shot.
- So với bộ lọc retrieval:
  - selector chọn ít tool hơn khoảng 3.4 lần nếu lấy cả app (5.8 so với 19.6), và khoảng 8.6 lần nếu chỉ lấy đúng tool (2.3);
  - đổi lại, tỉ lệ mẫu thiếu tool cần thiết tăng từ 3% lên 11% (theo app) hoặc 20% (theo tool).
- Nếu dùng làm bộ lọc thì nên lọc theo app, gộp nhiều biến thể, và thêm app đã dùng trong history.
- Trước đó, bộ lọc app "oracle" (app gold) không làm tăng điểm agent (60.85, Δ −1.93 ± 1.90; xem `llm-hypotheses-2026-10-28.md`). Vì vậy giá trị chính của selector có thể nằm ở tốc độ hoặc ở định hướng cho agent, chứ chưa chắc ở điểm số.
