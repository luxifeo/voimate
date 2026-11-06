# Prompt planner: input là history + danh sách tool, output là bản kế hoạch (2026-10-29)

## Thiết lập

| Thành phần | Mô tả |
|---|---|
| Code | `lifemate/planner.py` (prompt `PLAN_RULES`, `plan_request`), `scripts/eval_planner.py` (đo) |
| Model | Qwen3.5-4B trên vLLM đang host, temperature 0, **không thinking**, `max_tokens` 1024 |
| Input (system) | `PLAN_RULES` + app đã cài, mỗi tool gồm: loại read/write, mô tả, tham số (`*` = bắt buộc), tham số ID ghi kèm read tool trả ra ID đó (`chat_id* <- chat_ott.list_chats`), và kiểu trả về của read tool |
| Input (user) | history (lời người dùng, tool call kèm kết quả cắt ở 1500 ký tự, lời trợ lý) + câu yêu cầu cuối |
| Output | Kế hoạch 6 mục: 1 Request (Heard / Changes / Meaning), 2 Context (Done / Known / Refers to), 3 Parts (do / cancelled / if + Scope), 4 Steps (`S1: tool \| mục đích \| tham số = giá trị (nguồn)`), 5 Missing information (`ask: tool \| params`), 6 Tools |
| Đo | Tool trong plan = tool của các dòng `S<n>:` và `ask:`, so với tập tool gold (định nghĩa như `eval_tool_selector.py`). Dev 300, seed 0 |

Cách đo này chỉ kiểm phần chọn tool. Các phần khác của plan (nguồn tham số, ID, phần bị huỷ) chưa được chấm tự động.

## Kết quả qua các phiên bản prompt (dev 300, transcript ASR)

| Phiên bản | Thay đổi | Bao phủ tool | Đúng y tập tool | Ca bị cắt | s/mẫu |
|---|---|---|---|---|---|
| v3, thinking (`max_tokens` 4096) | | 44.7% | 36.7% | 95 | 2.19 |
| v3, không thinking | | 55.3% | 40.3% | 18 | 0.77 |
| v4, không thinking | giới hạn dòng "Fixes" (sửa lỗi ASR), cấm viết "wait / maybe / let's assume" | 58.0% | 41.3% | 4 | 0.39 |
| v5, không thinking | thêm 1 ví dụ mẫu, ví dụ dùng lại ID có sẵn trong history | 49.7% | 37.3% | 18 | 0.44 |
| **v6, không thinking** | bỏ dòng "Fixes" (đọc lỗi ASR ngay trong Meaning); ví dụ đổi sang có read tool chạy trước | **59.3%** | **44.3%** | 7 | 0.40 |
| v6, không thinking, **transcript gold** | | **78.3%** | **60.3%** | 3 | 0.38 |

Ghi chú:
- v3–v5 chạy bằng script nháp, rồi chấm lại bằng cùng cách đọc tool như `scripts/eval_planner.py`. v6 chạy bằng script này.
- [Inference] Chênh lệch giữa v4 và v6 (1.3 điểm bao phủ) nằm trong mức nhiễu của vLLM, chưa coi là cải thiện.

So với selector (`tool-selector-2026-10-29.md`):

| Selector / planner | Bao phủ (ASR) | Bao phủ (gold) |
|---|---|---|
| selector zero-shot, không thinking | 46.0% | – |
| selector zero-shot, thinking | 70.0% | 86.0% |
| selector 8-shot, thinking | 80.0% | – |
| planner v6, không thinking | 59.3% | 78.3% |

## Nhận xét

1. **Bật thinking làm planner kém hơn.** Ở v3, 95/300 ca bị cắt ở 4096 token, và 94 ca trong đó chưa kịp viết plan. Phần suy luận lặp đi lặp lại ở chỗ đoán lỗi ASR ("X is likely X or X …"). Selector chỉ trả JSON nên ít bị lặp hơn (10 ca).
2. **Mục sửa lỗi ASR gây lặp.** Ở v3 không thinking, 14/18 ca bị cắt kẹt ở dòng "Fixes". Ép model viết từng cặp sửa (v4) rồi bỏ hẳn dòng này (v6) làm số ca bị cắt giảm từ 18 xuống 4–7.
3. **Ví dụ mẫu dẫn hướng model rất mạnh.** Ví dụ v5 dùng lại ID có sẵn và không có read tool chạy trước. Kết quả là model bỏ read tool ở nhiều ca: so với v4, v5 mất bao phủ ở 44 ca và chỉ được thêm ở 19 ca. v6 đổi ví dụ sang có `chat_ott.list_chats` chạy trước và lấy lại được mức cũ.
4. **ASR vẫn là nguyên nhân lớn nhất:** transcript gold tăng bao phủ thêm 19 điểm (59.3% lên 78.3%). Các ví dụ cụ thể:
   - "tắt hộ chị đèn" bị ASR thành "viết tắt hồ chí đèn";
   - "dời sang" thành "rời sang", nên model chọn `ride_hailing`;
   - "hoãn hẳn, xoá luôn" thành "hoảng hẳn phát lưu".
5. **Lỗi còn lại của v6 (ASR):**

   | Loại lỗi | Số ca |
   |---|---|
   | Đúng y tập tool | 133 |
   | Thiếu write tool | 49 |
   | Chỉ chọn thừa | 36 |
   | Chỉ thiếu read tool | 26 |
   | Thiếu read tool và chọn thừa | 19 |
   | Trả rỗng dù gold cần tool | 22 |
   | Gold out-of-scope nhưng vẫn chọn tool | 8 |
   | Bị cắt | 7 |

   - Read tool hay bị thiếu: `contacts.search` 17, `calendar.list_events` 11, `banking.list_beneficiaries` 11.
   - Tool hay bị chọn thừa: `ride_hailing.get_quotes` 12, `banking.get_balance` 9.
   - Có ca hiểu ngược câu điều kiện: "nếu mai mưa thì …, không mưa thì thôi" thành "mưa thì không làm gì" (0328, 1628).

## Kết luận

- Với transcript ASR, prompt cho ra plan đủ 6 mục ở 288/300 mẫu với 0.4 s/mẫu. Nhưng nếu chỉ xét việc chọn tool, planner không thinking (59.3%) vẫn kém selector có thinking (70.0%).
- [Inference] Nếu dùng plan để dẫn agent, nên kết hợp với hai việc làm bằng code:
  - thêm read tool cần chạy trước theo 7 cặp đã kiểm chứng (`tool-selector-errors-2026-10-29.md`, mục 4);
  - gộp tool của plan với tool của selector.
  
  Chưa chạy thử.
- Giá trị thật của plan với agent (điểm state_hash và F1) chưa đo. Muốn biết thì phải đưa plan vào prompt agent và so sánh có cặp trên dev.
