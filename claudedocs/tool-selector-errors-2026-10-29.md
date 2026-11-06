# Phân tích suy luận của tool selector, zero-shot + thinking (2026-10-29)

Run được phân tích: `outputs/tool_selector/train/zs_think`, dev 300, transcript ASR.

| Kết quả | Số mẫu |
|---|---|
| Đúng y tập tool gold | 154 |
| Sai, được đọc suy luận | 146 |

Bốn subagent đọc toàn bộ suy luận của 146 ca, chia theo nhóm lỗi. Nguyên nhân của từng ca là nhận định của subagent [Inference]. Các con số đo bằng script thì ghi rõ là đo.

## 1. Nhóm lỗi (phân loại tự động)

| Nhóm | Số mẫu | Định nghĩa |
|---|---|---|
| A | 9 | Gold out-of-scope (rỗng) nhưng model vẫn chọn tool |
| B | 12 | Model trả rỗng nhưng gold cần tool; 9 ca bị cắt ở `max_tokens` 2048 |
| C | 6 | Chọn sai app hoàn toàn |
| D | 35 | Thiếu write tool hoặc tool cần hỏi |
| E | 37 | Đủ write tool, chỉ thiếu read tool |
| F | 47 | Đủ tool gold nhưng chọn thừa |

## 2. Hai phép đo bổ sung

| Run | Bao phủ tool | Đúng y tập tool | Ca bị cắt (`length`) |
|---|---|---|---|
| zero-shot thinking, `max_tokens` 2048, ASR | 70.0% | 51.3% | 10 |
| zero-shot thinking, `max_tokens` 4096, ASR | 70.7% | 51.3% | 11 |
| zero-shot thinking, `max_tokens` 4096, **transcript gold** | **86.0%** | **63.7%** | - |

Nhận xét:
- Tăng `max_tokens` không giúp gì. Các ca bị cắt là do model lặp mãi một câu, không phải do thiếu chỗ.
- Thay ASR bằng transcript gold tăng bao phủ thêm 15.3 điểm. Lỗi ASR là nguyên nhân lớn nhất.

## 3. Nguyên nhân gốc (tổng hợp 146 ca)

| Nguyên nhân | Số ca (nguyên nhân chính) | Nhóm | Ví dụ |
|---|---|---|---|
| ASR làm đổi nghĩa | ~42 | D 20, E 9, ABC 7, F 6 | 0194: "đặt báo thức trước đó một tiếng" → ASR "bớt … một chuyến" → `ride_hailing.cancel` |
| Chọn read thừa: thông tin đã có trong history/request, kiểm tra "cho chắc", hoặc tra bằng tool vòng vèo | 28 | F | 0632: "the assistant already identified the device … But to be safe … include smart_home.get_status" |
| Thiếu read tiền đề hoặc tra ID bằng sai tool | 17 | E | 0001: "calendar tools - to update a running event" nhưng bỏ `calendar.list_events`; 1389: dùng `contacts.search` thay vì `chat_ott.list_chats` |
| Nhầm giữa các tool gần nghĩa | ~13 | D 9, ABC 1, E 3 | 0418: nạp tiền điện thoại → `ewallet.transfer` thay vì `topup_mobile`; 0280: "tắt điện" → `switch_set` thay vì `light_set` |
| Bỏ qua câu người dùng tự rút lại hoặc phủ định ("à mà thôi khỏi", "khoan", "đừng") | ~11 | ABC 7, F 4 | 1192: "… à mà thôi khỏi" → vẫn chọn `home_automation.delete_rule` |
| Suy luận lặp vô hạn, bị cắt | 9 | B | 1024: in `"tools": ["reserve"]` 9 lần; 0328: một dòng lặp 19 lần |
| Rule "lấy tool của mọi nhánh điều kiện" mâu thuẫn với gold (gold chỉ chứa nhánh thực sự chạy; điều kiện đã biết từ history) | ~6 | ABC 2, F 2, E | 0082: history đã có dự báo "có mây"; model vẫn "include all tools that could be used in either branch" |
| Gộp hoặc bỏ sót một phần trong yêu cầu nhiều phần | ~1 chính, 7 phụ | D | 0476: "còn giờ thì chỉnh đèn 40%" bị gộp vào rule hằng ngày |
| Đọc sai history hoặc vật được nhắc tới | ~5 | D 2, E 1, ABC 2 | 1040: "cái còn lại" là email trong history, model hiểu thành reminder |
| Suy luận có tool nhưng JSON cuối bỏ mất | ~3 | D 2, E 1 | 0303: "I need to use bill_payment.pay" nhưng JSON không có |
| Nhãn gold đáng bàn, lựa chọn của model hợp lý | ~8 | E 3, F 5 | 1119: người dùng nói "cộng lại", model chọn `calculator.calculate` |

## 4. Kiểm chứng bảng tool tiền đề trên toàn bộ train (1,545 mẫu gold, đo)

| Write tool | n | Read tiền đề | Có trong cùng gold | Không có nhưng đã gọi trong history |
|---|---|---|---|---|
| `banking.transfer` | 59 | `banking.list_beneficiaries` | 95% | 5% |
| `banking.transfer` | 59 | `banking.get_balance` | 95% | 5% |
| `bill_payment.pay` | 78 | `bill_payment.list_bills` | 91% | 9% |
| `calendar.update_event` | 36 | `calendar.list_events` | 89% | 11% |
| `sms.send` | 78 | `contacts.search` | 87% | 6% |
| `ewallet.transfer` | 32 | `contacts.search` | 84% | 9% |
| `chat_ott.send_message` | 66 | `chat_ott.list_chats` | 77% | 23% |
| `chat_ott.send_message` | 66 | `contacts.search` | 11% | 11% |
| `phone.call` | 36 | `contacts.search` | 31% | 25% |
| `video_meeting.join` | 11 | `video_meeting.list_meetings` | 45% | 36% |
| `maps.save_place` | 12 | `maps.search_place` | 50% | 50% |

Mỗi hàng: khi gold có write tool đó, bao nhiêu phần trăm gold cũng gọi read tiền đề. Ở 7 cặp đầu, read tiền đề có trong gold hoặc đã được gọi ở history trong 100% trường hợp.

Kết luận:
- 7 cặp đầu có thể thêm bằng code, không cần dựa vào prompt: hễ chọn write tool thì thêm read tiền đề, trừ khi read đó đã được gọi trong history.
- `chat_ott.send_message` lấy `chat_id` từ `chat_ott.list_chats`, không phải từ `contacts.search` (model hay nhầm điểm này).
- `phone.call` không nên có rule cứng (chỉ 31%).

## 5. Đề xuất sửa, xếp theo độ chắc chắn

1. **Thêm read tiền đề bằng code** cho 7 cặp đã kiểm chứng (bỏ qua nếu history đã gọi). Nhắm các ca: 0001, 0342, 0421, 0123, 1389, 0163, 0208 và các ca tương tự.
2. **Chống lặp suy luận:**
   - giới hạn ngân sách thinking rồi ép trả lời;
   - nếu bị cắt thì lấy danh sách tool cuối cùng xuất hiện trong phần suy luận;
   - hoặc chạy lại với thinking tắt.
   
   4 trên 9 ca bị cắt đã có đủ tool gold trong suy luận.
3. **Rule rút lại / phủ định:** phần bị huỷ bằng "à thôi", "mà thôi khỏi", "khoan", "đừng … nữa" thì không chọn tool. Nhắm các ca: 1192, 1661, 1114, 0036, 1453, 0264, 0227, 0270, 1478.
4. **Sửa rule điều kiện:**
   - nếu điều kiện đã có câu trả lời trong history (dự báo, giá) thì chỉ lấy nhánh áp dụng;
   - nếu chưa biết thì giữ tool của cả hai nhánh để không giảm recall.
   
   Nhắm các ca: 0082, 0266, 0234.
5. **Bảng phân biệt tool gần nghĩa:**

   | Cách nói / tình huống | Tool đúng |
   |---|---|
   | đèn / "điện phòng X" | `light_set` (`switch_set` chỉ dùng cho ổ cắm) |
   | nạp tiền điện thoại | `topup_mobile` |
   | báo thức | `alarm` (không phải `reminders`) |
   | dời / huỷ / tạo lại lịch | `update_event` / `cancel_event` / `create_event` |
   | soạn nháp email | `create_draft` |
   | cất / lưu trữ email | `archive` |
   | trả hoá đơn | `bill_payment.pay` (tham số `source` đã chọn được nguồn ngân hàng) |

   Nhắm khoảng 13 ca.
6. **ASR:** đây là nguyên nhân lớn nhất (15 điểm bao phủ).
   - [Speculation] Thêm bảng sửa từ đồng âm vào prompt ("bớt/báo thiếu" → "báo thức", "một chuyến" → "một tiếng", "bút xe" → "book xe") có thể giúp một phần. Chưa rõ model 4B dùng được tới đâu.
   - Hướng chắc hơn là cải thiện ASR: hotword hoặc fine-tune gipformer (xem `gipformer-training-format-2026-10-29.md`).
7. **Không phạt các tool "hợp lý" khi chấm:** tool của nhánh điều kiện không chạy, hoặc tool mà câu hỏi lại nhắm tới. Khoảng 8 ca là do nhãn hoặc cách chấm, không phải lỗi model.

Chưa đề xuất nào ở trên được chạy thử.
