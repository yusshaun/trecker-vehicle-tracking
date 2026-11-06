# Vehicle Tracking with YOLOv11 and Depth Anything V2

以電腦視覺分析影片中的車輛，結合 YOLOv11 物件偵測、自製追蹤與移動判斷，以及 Depth Anything V2 深度估計。

## 功能

- 偵測汽車、機車、公車與卡車
- 以 IoU、中心距離、顏色直方圖、尺寸與移動方向維持追蹤 ID
- 判斷車輛為移動中或停止，並以綠框/紅框標示
- 以深度圖補充 2D 位置變化以外的遠近移動資訊
- 支援本地影片與 YouTube 影片串流
- GPU 可用時自動使用 CUDA 與 FP16

## 專案結構

```text
main.py                 # 主要偵測、追蹤與輸出流程
depth_integration.py    # Depth Anything V2 / Hugging Face 整合
record_video.py         # 從本地檔案或 YouTube 錄製測試影片
Depth-Anything-V2/      # Depth Anything V2 原始碼（Apache-2.0）
requirements.txt        # Python 依賴
```

## 環境需求

- Python 3.10+
- 建議使用 NVIDIA GPU 與相容的 PyTorch CUDA 版本
- Windows、Linux 或 macOS（GPU 加速依硬體與 PyTorch 安裝方式而異）

建立虛擬環境並安裝套件：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

> PyTorch 的 CUDA 安裝方式請依照 [官方安裝選擇器](https://pytorch.org/get-started/locally/) 選擇對應版本。

## 模型與測試影片

大型模型權重與影片不提交到 GitHub。執行前請準備：

1. 將 YOLO 權重 `yolo11l.pt` 放在專案根目錄，或修改 `main.py` 的 `model_path`。
2. 將測試影片放到 `test_video/recorded_15s.mp4`，或在 `main()` 中修改 `video_url`。
3. 如要使用本地 Depth Anything V2 模型，將 checkpoint 放到：
   `Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth`

若沒有本地深度 checkpoint，程式會嘗試使用 Hugging Face 的深度模型；第一次執行需要網路下載模型。

## 執行

```bash
python main.py
```

按 `q` 可停止處理。輸出影片預設寫入 `fin.mp4`（已被 `.gitignore` 排除）。

## 設計重點

追蹤器不只依賴單一幀的 YOLO 結果，而是將候選框與歷史特徵比對，降低遮擋或短暫漏檢造成的 ID 變更。移動判斷同時參考影像平面中的位移與深度變化，因此能辨識車輛接近或遠離鏡頭的情況。

## 致謝與授權

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 原始碼以 Apache License 2.0 授權，授權文件保留於 `Depth-Anything-V2/LICENSE`。
- YOLOv11 由 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供；請依其授權條款使用模型與套件。
- 測試影片與模型權重未納入本專案，使用時請確認來源與授權。
