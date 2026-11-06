# gipformer 1.5: cách huấn luyện và định dạng input (2026-10-29)

Mục tiêu: kiểm tra audio và cấu hình đưa vào gipformer có khớp với lúc model được huấn luyện không.

Nguồn:
- model card [g-group-ai-lab/gipformer1.5-68M-rnnt](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt);
- repo [ggroup-ai-lab/gipformer](https://github.com/ggroup-ai-lab/gipformer) (`infer_onnx.py`, `infer_pytorch.py`);
- Space [gipformer-demo](https://huggingface.co/spaces/g-group-ai-lab/gipformer-demo) (`app.py`, `chunking.py`, `Dockerfile`);
- file trong HF cache (`config.json`, `tokens.txt`, `model.pt`). `model.pt` chỉ được đọc bằng `pickletools`, không unpickle.

## 1. Những gì biết được về cách huấn luyện

| Hạng mục | Thông tin | Nguồn |
|---|---|---|
| Framework | icefall + k2, recipe `egs/librispeech/ASR/zipformer`, model Zipformer transducer 68.6M tham số | `infer_pytorch.py` (`get_model`, `add_model_arguments` của recipe này), `config.json` |
| Loss | transducer kèm nhánh CTC: checkpoint có `ctc_output`, `simple_am_proj`, `simple_lm_proj` | `model.pt` |
| Feature | kaldifeat fbank 80 bins, 16 kHz, `dither=0`, `snip_edges=False`, `high_freq=-400` | `infer_pytorch.py` |
| Decoder | stateless, `context_size=2` | `infer_pytorch.py` |
| Vocab | BPE 2000 (sentencepiece). Token viết hoa, không chữ số, không dấu câu; vài token tiếng Anh (`FACEBOOK`, `WEB`, `NEW`...) | `tokens.txt` |
| Nhãn huấn luyện | [Inference] chữ hoa, bỏ dấu câu, số viết bằng chữ. Suy ra từ vocab và từ chuẩn hoá WER trên model card ("lowercased, punctuation removed, and numbers converted to spoken form") | `tokens.txt`, model card |
| Dữ liệu huấn luyện, số giờ, augmentation (speed perturb, MUSAN, SpecAugment) | **Không công bố.** Checkpoint chỉ chứa `model`, không lưu tham số huấn luyện | model card, repo, `model.pt` |
| Cách dùng chính thức | ffmpeg đưa về 16 kHz mono; VAD Silero tách đoạn ≤ 20 s khi audio dài; fp32; `modified_beam_search` với 4 nhánh | `app.py`, `chunking.py`, `infer_onnx.py` |

## 2. Đối chiếu với pipeline hiện tại

| Hạng mục | Cách chính thức | Pipeline hiện tại | Khớp? |
|---|---|---|---|
| Sample rate, số kênh | 16 kHz mono | toàn bộ 6,969 file train + public là WAV PCM16 16 kHz mono | khớp |
| Dải giá trị mẫu | float [-1, 1] | `soundfile` float32 [-1, 1], sherpa `normalize_samples=True` | khớp |
| Fbank | xem mục 1 | sherpa `FeatureExtractorConfig`: 80 bins, `low_freq=20`, `high_freq=-400`, `dither=0`, `snip_edges=False` | khớp |
| Độ dài clip | ≤ 20 s mỗi đoạn | clip dài khoảng 7 s | khớp, không cần VAD |
| Trọng số | fp32 | `encoder.onnx` fp32 | khớp |
| Decode | `modified_beam_search`, 4 nhánh | `greedy_search` | **lệch** |
| Mức tín hiệu | không nêu | peak khoảng −7.7 dBFS, RMS khoảng −26 dBFS, DC ≈ 0 | không có dấu hiệu bất thường |

## 3. Đo decode (dev 983 clip, `scripts/bench_decode.py`)

| Biến thể | WER all | final | history | clip tốt hơn / tệ hơn | text giống production | s/clip (8 thread) |
|---|---|---|---|---|---|---|
| greedy, batch 32 (production) | 17.12 | 16.80 | 17.42 | - | 983 | 0.152 |
| greedy, batch 1 | 17.13 | 16.80 | 17.44 | 4 / 6 | 969 | 0.103 |
| beam 4 (chính thức) | 17.02 | 16.85 | 17.18 | 51 / 43 | 834 | 0.169 |
| beam 8 | 17.00 | 16.80 | 17.18 | 47 / 36 | 847 | 0.199 |

Đánh giá:
- Beam search giảm WER 0.10–0.12 điểm. Số clip tốt lên và xấu đi gần cân bằng (51 so với 43), nên có thể chỉ là nhiễu. Ở lượt cuối, chỉ số quan trọng nhất, beam không cải thiện.
- Decode theo lô 32 hay từng clip chỉ làm khác 14/983 transcript, WER không đổi.

## 4. Chênh lệch định dạng số

- Nhãn `audio_text` của cuộc thi có chữ số ở 98/983 clip (119 trên 13,736 từ). gipformer không thể in chữ số.
- WER của nhóm clip có chữ số là 27.03, nhóm không có là 15.88.
- Ví dụ: `8 giờ kém 15` ↔ `tám giờ ... mười lăm`, `HTV7` ↔ `htv bảy`, `RON95` ↔ `chín lăm`.
- [Inference] Mỗi chữ số được tính khoảng 1–2 lỗi. Vì vậy WER 17.12 bị đội lên khoảng 1–1.5 điểm do khác định dạng chứ không do nhận sai. Model card chuẩn hoá số về dạng đọc trước khi tính WER chính vì lý do này.
- Với agent, số dạng chữ (`sáu giờ`, `chín lăm`) thường vẫn hiểu được. Lỗi thật nằm ở từ bị nghe sai (`báo thức` → `vào thức`, `nhiêu` → `như`) và ở tên riêng hoặc từ tiếng Anh (68 từ chứa f/j/w/z trong reference, như `pizza`, `zalo`, `review`, `andrew`).

## 5. Kết luận

- Input đã đúng định dạng huấn luyện. Không cần resample, đổi kênh, chuẩn hoá mức tín hiệu hay tách VAD.
- Điểm lệch duy nhất là cách decode. Đổi sang beam 4 chỉ giảm khoảng 0.1 WER, nhiều khả năng là nhiễu, và tăng khoảng 0.02 s/clip. Bản thân nó không đáng đổi, nhưng beam search là điều kiện cần cho hotword biasing (mục 6).
- Muốn ASR tốt hơn phải đổi những gì model nhận ra, không phải định dạng input.

## 6. Hướng tiếp theo (chưa làm)

1. **Hotword biasing.** sherpa-onnx hỗ trợ `hotwords_file` với `modified_beam_search` và `modeling_unit="bpe"` + `bpe_vocab`. Có thể đẩy xác suất các từ trong tên app, tên tool và mô tả trong catalog của split. Catalog là input hợp lệ; không đọc `<split>_environment/`.
2. **Fine-tune gipformer trên audio train.** Có khoảng 10 giờ audio có nhãn (5,036 clip × khoảng 7.2 s), và checkpoint icefall (`model.pt`) đi kèm model. Nhãn cần chuẩn hoá về đúng định dạng huấn luyện (chữ hoa, bỏ dấu câu, số viết bằng chữ). Số tham số không đổi.
