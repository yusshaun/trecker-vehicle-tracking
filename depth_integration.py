"""
深度整合模組：封裝 Depth-Anything-V2 的模型載入與推論，
並提供在 bounding box 上計算平均深度的工具函式。

此模組會嘗試從專案內的 `Depth-Anything-V2` 目錄找到模型程式與 checkpoint。
若專案結構不同，也會順便嘗試舊版的 `../depth/Depth-Anything-V2` 路徑。
"""
import os
import sys
import numpy as np
import torch
import cv2

try:
    from transformers import pipeline
    from PIL import Image
except Exception:
    pipeline = None
    Image = None

def _resolve_depth_anything_base_dir():
    """尋找 Depth-Anything-V2 原始碼目錄。"""
    current_dir = os.path.dirname(__file__)
    candidates = [
        os.path.abspath(os.path.join(current_dir, 'Depth-Anything-V2')),
        os.path.abspath(os.path.join(current_dir, '..', 'depth', 'Depth-Anything-V2')),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return candidates[0]


BASE_DIR = _resolve_depth_anything_base_dir()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except Exception as e:
    DepthAnythingV2 = None
    print(f"Warning: 無法 import DepthAnythingV2: {e}")


class DepthEstimator:
    """載入模型並針對整張影像推論深度，提供在 bbox 上計算平均深度的方法。"""
    def __init__(self, encoder='vitl', checkpoint_path=None, device=None, input_size=518):
        self.encoder = encoder
        self.input_size = input_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.backend = None
        self.hf_pipe = None

        if DepthAnythingV2 is None:
            print('DepthAnythingV2 模組不可用，將嘗試 Hugging Face 深度模型')

        if checkpoint_path is None:
            checkpoint_path = os.path.join(BASE_DIR, 'checkpoints', f'depth_anything_v2_{encoder}.pth')

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }

        if encoder not in model_configs:
            print(f'Unsupported encoder: {encoder}, 深度功能停用')
            return

        cfg = model_configs[encoder]

        if DepthAnythingV2 is not None and os.path.isfile(checkpoint_path):
            try:
                self.model = DepthAnythingV2(**cfg)
                state = torch.load(checkpoint_path, map_location='cpu')
                self.model.load_state_dict(state)
                self.model = self.model.to(self.device).eval()
                self.backend = 'local'
                print(f'✓ 已載入深度模型: {checkpoint_path} (device={self.device})')
                return
            except Exception as e:
                print(f'Warning: 無法載入深度模型 checkpoint: {e}')
                self.model = None

        if pipeline is not None and Image is not None:
            try:
                hf_model_map = {
                    'vits': 'depth-anything/Depth-Anything-V2-Small-hf',
                    'vitb': 'depth-anything/Depth-Anything-V2-Base-hf',
                    'vitl': 'depth-anything/Depth-Anything-V2-Large-hf',
                    'vitg': 'depth-anything/Depth-Anything-V2-Large-hf',
                }
                hf_model_id = hf_model_map.get(encoder, hf_model_map['vitl'])
                torch_dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
                self.hf_pipe = pipeline(
                    task='depth-estimation',
                    model=hf_model_id,
                    device=0 if self.device.startswith('cuda') else -1,
                    dtype=torch_dtype,
                )
                self.backend = 'hf'
                print(f'✓ 已切換到 Hugging Face 深度模型: {hf_model_id} (device={self.device})')
                return
            except Exception as e:
                print(f'Warning: Hugging Face 深度模型初始化失敗: {e}')

        print('Depth-Anything-V2 / Hugging Face 深度功能都不可用')

    def infer_frame(self, frame, input_size=None):
        """對整張 BGR 影像回傳深度地圖 (numpy array)，失敗回傳 None。"""
        if self.backend is None:
            return None
        if input_size is None:
            input_size = self.input_size

        try:
            if self.backend == 'local' and self.model is not None:
                depth = self.model.infer_image(frame, input_size=input_size)
                return depth

            if self.backend == 'hf' and self.hf_pipe is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                result = self.hf_pipe(image)
                depth = result.get('depth') if isinstance(result, dict) else result
                return np.asarray(depth)

            return None
        except Exception as e:
            print(f'Warning: depth inference failed: {e}')
            return None

    @staticmethod
    def get_mean_depth(depth_map, box):
        """計算 depth_map 中 box 區域的平均深度，若無效則回傳 None。"""
        if depth_map is None:
            return None
        h, w = depth_map.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        return float(np.mean(roi))
