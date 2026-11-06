# Speech enhancement trước ASR gipformer (2026-10-28)

Script: `scripts/bench_enhance.py` (script mới, không sửa `lifemate/*.py`, `scripts/predict.py`, Docker).
Cache transcript: `outputs/asr/train_gip15_<method>.json` (định dạng `{rel_path: text}`, dùng với `--asr-cache _gip15_<method>`).
Số liệu phụ: `outputs/asr/enhanced/` (`timing.json`, `snr_user_audio.json`, `htdemucs_residual_db.json`, `logs/`).

## Cách đo

- Mỗi phương pháp: audio → enhancement → cùng `TransducerASR` (gipformer 1.5, sherpa-onnx 2.0.1, greedy, 8 thread). Chạy lại raw trên 983 clip dev: 0/983 khác `train_gip15.json`.
- Dev 300 = `random.Random(0).sample(load_split("train"), 300)`: 300 final + 683 history = 983 clip.
- Held-out: final turn của 1245 row ngoài dev; và mọi user audio có SNR < 20 dB của 1245 row đó (638 clip, trong đó 281 clip < 15 dB).
- WER/CER: corpus-level, cùng chuẩn hoá với `lifemate.asr.error_rates`. Cột num-norm dùng `bench_asr.normalize_vi`. CI 95%: paired bootstrap theo clip (2000 lần).
- SNR: cùng công thức với `audio_stats.json` (percentile 95 − percentile 10 của năng lượng frame 25 ms). Tính lại cho cả history; trên 1545 final, sai khác so với `audio_stats.json` ≤ 1.6e-5 dB.
- `_mixNN`: `α·enhanced + (1−α)·raw`, α = NN/100. Lag giữa enhanced và raw = 0 mẫu cho cả 3 model (đo bằng cross-correlation).
- `_warm`: đưa 2 s đầu clip (đảo ngược) vào trước để làm ấm state của model streaming, sau đó cắt bỏ phần này.
- `_gateT`: dùng transcript enhanced nếu SNR < T dB, còn lại dùng raw.

## 1. Kết luận

1. Không phương pháp nào giảm WER tổng một cách có ý nghĩa thực tế.
2. Dùng tín hiệu enhanced thuần làm WER tăng (dev, mọi clip): GTCRN +2.72, DPDFNet2 +2.50, DPDFNet8 +12.75, htdemucs +0.25 điểm.
3. Trộn 50–70% với audio gốc loại bỏ phần lớn thiệt hại, nhưng không tạo lợi ích. Biến thể tốt nhất trên dev (`dpdfnet8_mix50`, −0.08 [−0.33, +0.16]) đảo chiều trên held-out final (+0.16 [+0.01, +0.30]).
4. Tín hiệu dương duy nhất: htdemucs (stem vocals) trên clip SNR < 15 dB. Trên 329 clip này (dev + held-out): −0.79 điểm WER [−1.35, −0.24]. Riêng held-out: −0.85 [−1.43, −0.29].
5. Gate "htdemucs nếu SNR < 15 dB" trên toàn bộ 5036 user audio của train: WER 16.74% → 16.69% (−0.046 điểm [−0.079, −0.014]). Mức giảm đo được, nhưng rất nhỏ.
6. WER của gipformer gần như không phụ thuộc SNR: raw 20.0% (< 15 dB) so với 17.0% (≥ 40 dB). Phần lớn lỗi ASR không do nhiễu.
7. Khuyến nghị: không đưa enhancement vào pipeline (mục 7).

## 2. WER/CER, dev 300 (300 final + 683 history)

| method | final WER | final CER | hist WER | hist CER | all WER | all WER num-norm | ΔWER all vs raw [95% CI] |
|---|---|---|---|---|---|---|---|
| raw | 16.80% | 10.30% | 17.42% | 10.68% | 17.12% | 15.10% | — |
| gtcrn | 18.95% | 12.00% | 20.68% | 13.18% | 19.85% | 17.90% | +2.72 [+2.25, +3.22] |
| gtcrn_mix70 | 17.42% | 10.73% | 17.80% | 10.98% | 17.62% | 15.60% | +0.50 [+0.23, +0.76] |
| gtcrn_mix50 | 17.19% | 10.61% | 17.69% | 10.83% | 17.45% | 15.45% | +0.33 [+0.14, +0.53] |
| gtcrn_warm | 18.62% | 11.68% | 19.83% | 12.45% | 19.25% | 17.29% | +2.13 [+1.68, +2.57] |
| gtcrn_warm_mix50 | 17.25% | 10.63% | 17.67% | 10.77% | 17.47% | 15.46% | +0.35 [+0.16, +0.54] |
| dpdfnet2 | 18.83% | 11.96% | 20.37% | 13.21% | 19.63% | 17.68% | +2.50 [+2.01, +3.01] |
| dpdfnet2_mix70 | 17.12% | 10.54% | 17.70% | 10.87% | 17.42% | 15.43% | +0.30 [+0.05, +0.54] |
| dpdfnet2_mix50 | 17.01% | 10.45% | 17.45% | 10.73% | 17.24% | 15.23% | +0.12 [−0.08, +0.32] |
| dpdfnet2_warm | 19.59% | 12.52% | 22.13% | 14.62% | 20.91% | 18.98% | +3.79 [+3.20, +4.38] |
| dpdfnet2_warm_mix50 | 17.07% | 10.46% | 17.80% | 10.90% | 17.45% | 15.42% | +0.33 [+0.11, +0.55] |
| dpdfnet8 | 25.82% | 18.78% | 33.62% | 26.21% | 29.87% | 28.07% | +12.75 [+11.34, +14.21] |
| dpdfnet8_mix50 | 16.97% | 10.47% | 17.11% | 10.48% | 17.04% | 15.02% | −0.08 [−0.33, +0.16] |
| htdemucs | 17.12% | 10.55% | 17.62% | 10.79% | 17.38% | 15.35% | +0.25 [−0.01, +0.58] |
| htdemucs_mix70 | 16.86% | 10.38% | 17.45% | 10.63% | 17.17% | 15.14% | +0.04 [−0.11, +0.18] |
| htdemucs_mix50 | 16.86% | 10.36% | 17.48% | 10.68% | 17.18% | 15.16% | +0.06 [−0.05, +0.16] |
| htdemucs_gate15 | 16.71% | 10.25% | 17.46% | 10.67% | 17.10% | 15.08% | −0.02 [−0.12, +0.04] |
| htdemucs_mix70_gate15 | 16.72% | 10.25% | 17.44% | 10.66% | 17.09% | 15.07% | −0.03 [−0.12, +0.03] |
| dpdfnet2_mix50_gate15 | 16.68% | 10.26% | 17.42% | 10.67% | 17.06% | 15.04% | −0.06 [−0.16, +0.02] |

Held-out và toàn bộ train (final turn):

| method | held-out final (1245) WER | ΔWER [95% CI] | toàn bộ 1545 final WER | CER |
|---|---|---|---|---|
| raw | 16.93% | — | 16.90% | 10.59% |
| dpdfnet8_mix50 | 17.09% | +0.16 [+0.01, +0.30] | 17.06% | 10.73% |
| htdemucs | 16.95% | +0.02 [−0.08, +0.13] | 16.98% | 10.68% |
| htdemucs_mix70 | 16.91% | −0.02 [−0.11, +0.06] | 16.90% | 10.61% |
| htdemucs_gate15 | 16.88% | −0.04 [−0.09, −0.01] | 16.85% | 10.58% |
| dpdfnet2_mix50_gate15 | 16.90% | −0.03 [−0.07, +0.00] | 16.86% | 10.57% |

Toàn bộ user audio của train (1545 final + 3491 history; gồm cả 300 row dev):

| method | final WER | hist WER | all WER | all CER | ΔWER all [95% CI] |
|---|---|---|---|---|---|
| raw | 16.90% | 16.58% | 16.74% | 10.31% | — |
| htdemucs_gate15 | 16.85% | 16.54% | 16.69% | 10.28% | −0.046 [−0.079, −0.014] |
| htdemucs_gate20 | 16.87% | 16.55% | 16.71% | 10.28% | −0.029 [−0.070, +0.011] |
| htdemucs_mix70_gate15 | 16.87% | 16.54% | 16.70% | 10.28% | −0.037 [−0.066, −0.010] |
| dpdfnet2_mix50_gate15 | 16.86% | 16.57% | 16.71% | 10.28% | −0.029 [−0.061, +0.000] |

## 3. WER theo SNR

Phân bố SNR (5036 user audio của train): final p10 = 19.7 dB, 160/1545 clip < 20 dB. History có đuôi thấp nặng hơn: 594/3491 clip < 20 dB, 264 clip < 15 dB.

Dev 300, mọi clip:

| method | < 15 dB | 15–20 | 20–25 | 25–30 | 30–40 | ≥ 40 dB |
|---|---|---|---|---|---|---|
| số clip | 48 | 68 | 55 | 75 | 216 | 521 |
| raw | 20.0% | 13.8% | 19.8% | 17.1% | 17.2% | 17.0% |
| gtcrn | 23.4% | 15.1% | 22.7% | 21.3% | 20.6% | 19.2% |
| gtcrn_mix50 | 20.3% | 14.1% | 20.1% | 17.6% | 17.7% | 17.2% |
| dpdfnet2 | 20.8% | 14.7% | 21.6% | 21.1% | 20.1% | 19.5% |
| dpdfnet2_mix50 | 18.8% | 14.3% | 19.7% | 17.6% | 17.3% | 17.2% |
| dpdfnet8 | 20.6% | 14.8% | 27.4% | 29.1% | 33.4% | 31.4% |
| dpdfnet8_mix50 | 18.5% | 14.0% | 19.0% | 17.6% | 17.4% | 16.9% |
| htdemucs | 19.5% | 13.8% | 19.6% | 17.1% | 17.8% | 17.3% |
| htdemucs_mix70 | 19.4% | 14.0% | 19.4% | 16.9% | 17.4% | 17.1% |

Kiểm tra trên held-out (giả thuyết lấy từ dev: lợi ích chỉ có ở clip SNR thấp):

| method | held-out WER SNR<15 (n=281) | ΔWER [95% CI] | pooled dev+held-out SNR<15 (n=329) | ΔWER [95% CI] |
|---|---|---|---|---|
| raw | 19.73% | — | 19.77% | — |
| dpdfnet2_mix50 | 19.36% | −0.37 [−0.91, +0.14] | 19.27% | −0.50 [−1.05, +0.02] |
| dpdfnet2_mix70 | 19.36% | −0.37 [−1.00, +0.23] | 19.36% | −0.41 [−1.01, +0.17] |
| htdemucs | 18.88% | −0.85 [−1.43, −0.29] | 18.98% | −0.79 [−1.35, −0.24] |
| htdemucs_mix70 | 19.08% | −0.65 [−1.19, −0.16] | 19.13% | −0.64 [−1.15, −0.14] |

Ở dải 15–20 dB, htdemucs không giúp: nâng ngưỡng gate từ 15 lên 20 dB làm kết quả kém đi (bảng mục 2).

Trên dev, khi phân nhóm theo năng lượng phần "không phải vocals" mà htdemucs bỏ đi, nhóm bị bỏ nhiều nhất (residual > −10 dB, 41 clip) vẫn tệ hơn: raw 19.7%, htdemucs 23.2%.

Lưu ý: kiểm tra held-out so sánh 4 phương pháp × 2 ngưỡng, nên có rủi ro so sánh nhiều lần. Ngưỡng 15 dB được chọn sau khi xem dev.

## 4. Thời gian chạy mỗi clip

Đo tuần tự từng clip trong một process (`bench_enhance.py timing`), clip dev, audio trung bình 5.17 s. CPU dùng chung với tiến trình khác: load average 18–21 trên 32 core khi đo; lúc load ~50, htdemucs CPU mất 15–33 s/clip. Vì vậy số CPU là cận trên, không phải số trên máy chấm.

| model | thiết bị | mean s/clip | p95 s/clip | RTF |
|---|---|---|---|---|
| GTCRN | CPU, 8 thread | 0.49 | 0.84 | 0.095 |
| DPDFNet2 | CPU, 8 thread | 1.38 | 2.58 | 0.27 |
| DPDFNet8 | CPU, 8 thread | 3.51 | 6.37 | 0.68 |
| htdemucs | CPU, 8 thread (10 clip) | 2.66 | 4.22 | 0.54 |
| htdemucs | GPU1 A30 (dùng chung), 40 clip | 0.30 | 0.41 | 0.058 |

- htdemucs trên GPU: peak `torch.cuda.max_memory_allocated` = 570 MB (chưa tính CUDA context).
- GTCRN/DPDFNet chạy qua sherpa-onnx pip, chỉ đo trên CPU.
- So sánh: ASR gipformer raw ≈ 0.31 s/clip (batch 32, 8 thread, load ~50).
- Mỗi sample có trung bình 3.26 user audio (train) và 3.05 (public_test), tối đa 13. [Inference] Gate htdemucs < 15 dB trên GPU chạy cho ~6.5% clip, thêm khoảng 0.06 s/sample trung bình. Trường hợp xấu nhất (13 audio đều bị gate, p95 0.41 s) thêm ~5 s. [Inference] Trên CPU (2.66 s/clip, đo khi máy tải nặng), trường hợp xấu nhất có thể vượt 30 s.

## 5. Model đã dùng

| model | tham số | phát hành / nguồn | licence |
|---|---|---|---|
| GTCRN (`gtcrn_simple.onnx`, sherpa-onnx) | 82,580 phần tử initializer trong ONNX (gồm cả hằng số) | repo Xiaobin-Rong/gtcrn tạo 2023-09-01; ONNX trong release `speech-enhancement-models` của sherpa-onnx, 2025-03-10; sha256 khớp `checksum.txt` | MIT |
| DPDFNet2 / DPDFNet8 (CEVA, họ DeepFilterNet2 + dual-path RNN) | 2.51M / 3.56M phần tử initializer | HF `Ceva-IP/DPDFNet` tạo 2025-12-17; ONNX trong release sherpa-onnx, 2026-03-16 | Apache-2.0 |
| htdemucs (Demucs v4, weights `955717e8`) | 41,984,456 tham số (torch) | weights từ `dl.fbaipublicfiles.com/demucs`; package `demucs` 4.1.0 trên PyPI, 2026-07-11. [Unverified] htdemucs công bố năm 2022 (không kiểm tra lại trong phiên này) | MIT |

Mọi model chạy offline sau khi tải. Nếu thêm htdemucs: 4.5B + 68M + 568M + 42M ≈ 5.18B < 6B.

## 6. Ví dụ (htdemucs thuần, dev; số lỗi từ raw → enhanced)

htdemucs không đổi transcript ở 915/983 clip, giúp 27 clip, làm hại 41 clip.

Giúp:

| clip | SNR | ref | raw | htdemucs |
|---|---|---|---|---|
| train_1379 (12→7) | 11.3 | Anh về muộn chưa ăn gì, gọi giùm anh một tô ramen gà ở Osaka Ramen, trả bằng ví nha. | Đến về mùa chưa ăn gì gọi giúp mua toramen ra ở ô sakaramen trả bằng bí nga | Anh về muộn chưa ăn gì gọi dùng ít một tô ramen ga ở sakaramen trả bằng bí nga |
| train_1687_u4 (7→3) | 49.8 | Thêm việc gọi lại cho anh Minh lúc bốn giờ chiều nay. | Em việt minh lúc bốn giờ chiều mai | Em việc cho anh minh lúc bốn giờ chiều nay |
| train_1628_u7 (4→2) | 41.2 | Đặt hai bát bún bò đặc biệt với một chả cua, giao lên công ty nhé. | Đã khai bát bún bò đặc biệt với một gia cua ra lên công ty nhé | Đặt hai bát bún bò đặc biệt với một gia cua ra lên công ty nhé |
| train_1617_u1 (2→0) | 37.6 | Sáng nay Sài Gòn trời sao em? | Sáng nay sài gòn em | Sáng nay sài gòn trời sao em |
| train_1168 (4→3) | 39.6 | … hẹn bưu tá lấy nha … | … hẹn bu tá lấy nha … | … hẹn bưu tá lấy nha … |

Hại:

| clip | SNR | ref | raw | htdemucs |
|---|---|---|---|---|
| train_1151_u1 (0→13) | 52.6 | Gần trường em có quán cà phê sách nào yên tĩnh không ạ? | Gần trường em có quán cà phê sách nào yên tĩnh không ạ | Ok |
| train_1151 (7→19) | 38.2 | Tuần sau em ôn thi cuối kỳ nên cần chỗ ngồi học. … | Tuần sau em em thiếp trường không cần chỗ ngồi học … | Tuần sau em cũng có tiền không cần chỗ là muốn đi xe buýt … |
| train_1099 (3→6) | 37.0 | Tôi đang bốc hàng nè, … | Tôi đang bốc hàng nè … | Tôi bị bắt tôi đang bốc hàng nè … |
| train_1558 (1→3) | 69.3 | … bật lại chế độ vắng nhà … | … bật lại cái độ vắng nhà … | … bật lại cái đồ bánh nhà … |
| train_1618_u8 (0→2) | 69.6 | Thêm cái ba giờ chiều, đón thằng Bi. | Thêm cái ba giờ chiều đón thằng bi | Em kế ba giờ chiều đón thằng bi |

Lỗi điển hình của denoiser:

- GTCRN và DPDFNet làm mất phần đầu câu. Ví dụ train_0263, GTCRN: "Đặt báo thức năm rưỡi sáng mai…" → "Đã tích nay có chuyện ví…" (1 → 12 lỗi). DPDFNet2 hạ đoạn 0.5–1.0 s của clip này từ −38 dB xuống −82 dB.
- Biến thể `_warm` không sửa được lỗi này cho DPDFNet2. [Inference] Model coi một phần tiếng nói là nhiễu; nguyên nhân không chỉ là state khởi động lạnh.
- DPDFNet8 xoá gần hết tiếng nói ở 108/983 clip (mất ≥ 50% số từ), kể cả clip sạch. Ví dụ train_0683, SNR 40.6 dB: "Tạo sẵn giúp tôi một phòng họp trực tuyến…" → "Đây". [Unverified] Chưa loại trừ khả năng lỗi nằm ở cách sherpa-onnx 2.0.1 chạy DPDFNet8.

## 7. Khuyến nghị

1. Không thêm enhancement hay vocal separation vào pipeline.
   - Lợi ích lớn nhất đo được là −0.05 điểm WER trên toàn bộ user audio (gate htdemucs < 15 dB). [Inference] Mức này khó làm thay đổi điểm agent.
   - Chi phí: thêm dependency torch + demucs vào image Docker (uv 0.6.6 không lock được demucs, xem mục 8), thêm 42M tham số, chiếm GPU cạnh vLLM, thêm độ trễ. Separator đôi khi xoá cả câu (train_1151_u1 → "Ok").
2. Nếu vẫn muốn kiểm tra ở mức agent, cache đã sẵn sàng và phủ cả 5036 user audio của train: `--asr-cache _gip15_htdemucs_gate15`, so với `--asr-cache _gip15` trên dev 300.
3. Không dùng enhanced thuần trong bất kỳ trường hợp nào (GTCRN, DPDFNet, htdemucs đều làm WER tăng).
4. [Inference] Lỗi ASR còn lại phần lớn không do nhiễu: WER ở clip ≥ 40 dB vẫn là 17.0%. Các lỗi trong ví dụ là tên riêng, từ mượn và từ tiếng Anh (ramen, Osaka, bolero, SMS). Hướng có thể hiệu quả hơn là hotword/contextual biasing hoặc ASR có ngữ cảnh; hướng này chưa được đo.

## 8. Bỏ qua, thất bại, ghi chú

- **DeepFilterNet3: bỏ qua.**
  - `deepfilternet` 0.5.6 ghim `packaging<24` và `numpy<2`, trong khi lock dự án có packaging 26.3 và numpy 2.4.6.
  - Gói này cần `torchaudio`; index cu126 chỉ có tới torchaudio 2.11, còn torch của dự án là 2.14.1.
  - Thay bằng DPDFNet (họ DeepFilterNet, có sẵn trong sherpa-onnx, không thêm dependency).
- **Cài demucs:**
  - `uv add --group enhance demucs` thất bại với uv 0.6.6: resolver tách nhánh darwin-x86_64 do demucs ghim `torch<2.3` trên nền này, kể cả khi thêm marker `sys_platform == 'linux'`.
  - Đã chạy bằng overlay `uv run --with demucs==4.1.0` (nằm trong uv cache, không vào dự án).
  - Để chạy GPU: dùng torch cu126 của dự án, đặt `PYTHONPATH` chỉ trỏ tới demucs/einops/julius/lameenc/sphn (overlay có torch cu130, driver 12.2 không chạy được).
  - `pyproject.toml` và `uv.lock` không đổi (md5 trùng bản sao lưu).
- **UVR-MDX-NET-Voc_FT (sherpa-onnx):**
  - Chỉ thử trên 2 clip, không benchmark: trên CPU đang tải nặng chạy ~2× thời gian thực; licence của weights UVR chưa rõ.
- **VoxLock (sherpa-onnx):** không dùng. File được tải lên ngày 2026-10-26, không thoả điều kiện "phát hành trước 2026-10-26".
- **OOM:**
  - Một process ASR bị OOM killer dừng (RSS 14.8 GB, lúc máy hết RAM). Đã chạy lại, mỗi α một process.
  - Peak RSS của process ASR tăng theo số clip: 4.1 GB / 508 clip, 5.7 GB / 638 clip, 11.4 GB / 1115–1245 clip.
  - [Inference] Có thể do arena của onnxruntime tăng theo các shape khác nhau. `predict.py` hiện dùng Gemma ASR nên Docker không bị ảnh hưởng. Cần kiểm tra lại nếu sau này transcribe cả split bằng `TransducerASR` trong một process.
- **GPU:** chỉ dùng GPU1 (cho htdemucs), không dùng GPU0. Audio enhanced đã xoá; ONNX của GTCRN/DPDFNet giữ ở `outputs/asr/enhanced/models/` (≈ 25 MB).

Tái tạo:

```bash
uv run python scripts/bench_enhance.py enhance --method dpdfnet2 --set dev --workers 8
uv run python scripts/bench_enhance.py asr --method dpdfnet2 --alpha 1 0.7 0.5 --set dev
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<dir chứa demucs,einops,julius,lameenc,sphn> \
  uv run python scripts/bench_enhance.py enhance --method htdemucs --set dev --workers 2 --device cuda:0
uv run python scripts/bench_enhance.py gate --method htdemucs --snr 15
uv run python scripts/bench_enhance.py report --examples htdemucs
```

Model ONNX: `https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/{gtcrn_simple,dpdfnet2,dpdfnet8}.onnx` → `outputs/asr/enhanced/models/`.

## 9. Làm chậm audio trước ASR (2026-10-29)

Script: `scripts/bench_tempo.py`. Clip set: dev 300 (983 clip). Audio được biến đổi bằng ffmpeg rồi đưa vào recognizer gipformer production. Transcript của `raw` trùng 983/983 với `outputs/asr/train_gip15.json`.

| Cách | WER all | final | history | clip tốt hơn / tệ hơn so với raw |
|---|---|---|---|---|
| raw | **17.12** | 16.80 | 17.42 | - |
| atempo 0.9 (giữ cao độ) | 17.63 | 17.47 | 17.79 | 154 / 195 |
| atempo 0.8 | 18.67 | 18.53 | 18.80 | 144 / 265 |
| atempo 0.7 | 19.48 | 19.03 | 19.90 | 139 / 296 |
| rubberband 0.8 | 19.73 | 19.31 | 20.12 | 149 / 325 |
| speed 0.9 (hạ cao độ) | 17.16 | 16.65 | 17.63 | 181 / 180 |

WER theo tốc độ nói (chia 3 nhóm bằng nhau theo số từ/giây của reference; ngưỡng 2.63 và 3.18 từ/s):

| Cách | chậm | vừa | nhanh |
|---|---|---|---|
| raw | 18.70 | 17.16 | 16.11 |
| atempo 0.9 | 18.92 | 17.30 | 17.08 |
| speed 0.9 | 19.40 | 16.96 | 15.92 |

Kết luận: làm chậm audio không giảm WER.

- Giãn thời gian giữ cao độ làm WER tăng 0.5–2.6 điểm. Mức tăng xảy ra ở cả nhóm nói nhanh.
- Làm chậm kèm hạ cao độ cho kết quả ngang raw: số clip tốt hơn và tệ hơn gần bằng nhau (181 / 180).
- Chỉ làm chậm các clip nói nhanh (ước lượng tốc độ từ transcript raw) tốt nhất được 17.06, so với raw 17.12. Ngưỡng này lại chọn trên chính tập đo, nên chưa đủ làm bằng chứng.
- Nhóm nói nhanh vốn có WER thấp nhất. Vì vậy tốc độ nói không phải nguồn lỗi chính.
