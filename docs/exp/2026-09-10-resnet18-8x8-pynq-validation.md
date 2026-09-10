# ResNet-18 forward：2×2 與 8×8 systolic array 加速比較

## 結果

以 README 記錄的 2×2 實體 forward 為基準，8×8 的實測時間由 **28,031.949 秒降至
4,021.503 秒**，約 **6.97× 加速**、**減少 85.65% 耗時**。
兩次均在 PYNQ-Z1 執行相同 ResNet-18 匯出模型與驗證輸入，三個輸出皆通過 host digest 比對。
本紀錄只比較 forward，不納入 load model 時間。

整理日期：2026-09-10；關聯 [Issue #57](https://github.com/yenhao-huang/npu-pynq/issues/57)。

## 實測比較

| 指標 | 2×2 基準 | 8×8 本次 | 比較 |
| --- | ---: | ---: | --- |
| Forward 時間（秒） | 28,031.949 | 4,021.503 | **6.97× 加速** |
| Forward 時間（分鐘） | 467.199 | 67.025 | 約 7 小時 47 分 → 1 小時 7 分 |
| 時間減少 | — | 24,010.447 秒 | **85.65%**，約省 6.670 小時 |
| Physical matrix jobs | 2,104,040 | 135,290 | 減為原本的 6.43%，job 數比約 15.55 : 1 |
| MAC count | 1,814,073,344 | 1,814,073,344 | 相同工作量 |
| Processing elements | 4 | 64 | 數量增加 16 倍 |
| 實體矩陣限制（M, N, K） | `[2, 2, 256]` | `[8, 8, 256]` | M/N tile 擴大，K 上限相同 |
| 輸出一致性 | 3 / 3 PASS | 3 / 3 PASS | 三個輸出 digest 相同 |

計算公式（保留原始 notebook 精度）：

```text
speedup = 28031.949092986004 / 4021.5025392200005
        = 6.970516x

time_reduction = (1 - 4021.5025392200005 / 28031.949092986004) * 100
               = 85.653860%
```

README 將 2×2 時間四捨五入為 `28,031.949 seconds`；本表與公式已核對舊 notebook 的原始值。

## 計時範圍與比較條件

- **2×2**：舊 notebook 第 16 個 cell 的 `ModelResult` 輸出，
  `metrics.elapsed_seconds = 28031.949092986004`；計時是 `NPUModelRuntime.run()` 的內部耗時。
- **8×8**：本次 notebook 第 15 個 cell（第 7 步）的
  `runtime_summary['elapsed_seconds'] = 4021.5025392200005`；計時用 `time.monotonic()`，
  包含 `NPUModelRuntime(physical, model)` 建構及 `.run()`。
- 兩者的模型與 overlay 都在 forward 前完成載入；forward 後的輸出 digest 比對和證據寫入也不在計時範圍內。
- 兩次使用同一模型 manifest、payload、checkpoint、validation tensor 與 host acceptance，
  已逐項核對 SHA-256。輸入為 signed INT8、NHWC、`[1, 224, 224, 3]`、batch 1；
  模型有 49 個運算節點，包含 20 個 convolution 與 1 個 fully-connected。
- 兩筆資料是不同版本的單次 development 實體執行，並非同版程式、完全相同計時邊界的受控重複 benchmark。
  因此 **6.97× 是這兩次紀錄的 forward 實測比值**，不能將全部差異都歸因於 array 尺寸。

8×8 可在每次工作處理較大的輸出 tile，與 physical jobs 從 2,104,040 降到 135,290 的觀測相符。
但 PE 數量增加 16 倍並不等於端到端 forward 必然快 16 倍：Python runtime、資料重排、DMA、
控制流程及既有 CPU 算子仍有耗時。本次沒有各階段耗時拆解，不能量化它們各自的貢獻。

## 正確性與共同資料

兩次 `stem.relu`、`layer1.1.relu` 與 `logits` 都與相同的獨立 host reference 相符：

| 輸出 | Shape | SHA-256（兩次相同） |
| --- | --- | --- |
| `stem.relu` | `[1, 112, 112, 64]` | `a9a24748c2e4a589fd72589231f67c64b19c6bba4069e9ceb88ea2f95ed96404` |
| `layer1.1.relu` | `[1, 56, 56, 64]` | `b114c4775ae0cfc907aef33d2000be3fbacc69568182b24ee1a6839274f5a1d4` |
| `logits` | `[1, 1000]` | `2e2dea0c21c22c80beae0317d73b2f1459bb91f885a0b0eb2bb01bdc5142f936` |

共同 model manifest SHA-256：`698eb1148d62b0c2fee9b9db35f92eb0b3e6bb6c1c0de309dad0a26078790b9f`。

共同 validation tensor SHA-256：`94f8264f08bea2dcfab74adaf09e3bae0f950c68ab7d3b3dd02eb7315e00cb8c`。

共同 host acceptance SHA-256：`acf6ebec8a13695e7b2644f6e1f9408cfe3bdfba0047f695d4cf8ca9b87c6852`。

這是 synthetic、unlabeled validation input 的數值一致性與 forward 時間比較，沒有 ImageNet accuracy 結論。

## 版本與證據來源

| 項目 | 2×2 | 8×8 |
| --- | --- | --- |
| Deployment ID | `20260904-013134` | `resnet18-8x8-issue57-20260910` |
| 部署原始碼 commit | `03b8a071c45af6c1b86dc04dcbaf62705a251ec6` | `12fcef5c33a8ad4d0c6f1a651b668610df68bf6b` |
| Overlay artifact commit | `e1200ca804944ad446e47b46b16b169fa43c9312` | `d89cec902fb79eb1ef409e6749e196379a038db6` |
| Evidence class | `physical-pynq-z1-development` | `physical-pynq-z1-development` |

1. [README：Verified physical ResNet-18 run](https://github.com/yenhao-huang/npu-pynq/blob/95a759da2d86577d7d2f113a07eaa7204462b22a/README.md#verified-physical-resnet-18-run)：
   記錄 2026-09-04 的 2,104,040 jobs、28,031.949 秒與 3/3 digest PASS；同份 README 的硬體架構為 2×2。
2. [2×2 板端 notebook](http://192.168.2.99:9090/notebooks/npu_resnet18/releases/20260904-013134/examples/resnet18/resnet18.ipynb)：
   第 16 個 cell 保存精確 elapsed seconds，第 20 個 cell 保存 `[2, 2, 256]` 與比對證據。
3. [8×8 板端 notebook](http://192.168.2.99:9090/notebooks/npu_resnet18/releases/resnet18-8x8-issue57-20260910/examples/resnet18/resnet18.ipynb)：
   第 15 個 cell 保存 forward 時間與 jobs；使用者在最後一格確認 `human_approves = True` 並成功寫入 PASS。

本機原始快照位於忽略的 `build/exp-resnet18-20260910/`：

- [2×2 notebook](../../build/exp-resnet18-20260910/resnet18.2x2.executed.ipynb)，SHA-256：`b3c3dfa77621a8f1c63e201ab248b560208a9cc07878d8538b7735d3c6341213`。
- [8×8 notebook](../../build/exp-resnet18-20260910/resnet18.executed.ipynb)，SHA-256：`c347c899e2dd408b73acbadac8d7198f95de2de99e333ce3c35bd9dd3c739639`。
- [8×8 驗證 JSON](../../build/exp-resnet18-20260910/notebook-evidence-20250506T061451Z.json)，SHA-256：`cd1a523d7115f83cdf966d69a24edf5173d09fa824517be092f45de86b501c27`。

8×8 證據檔名中的板端 UTC 日期為 `2025-05-06T06:14:51Z`，與整理日期不一致；
保留原始證據名稱，不以板端日曆時間推定實際實驗日期。耗時使用 monotonic clock。
本次只讀取既有輸出與證據整理比較，沒有重新執行 forward，也沒有修改使用者 notebook。
