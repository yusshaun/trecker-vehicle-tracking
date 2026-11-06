"""
車輛追蹤系統 - 使用 YOLOv11 進行車輛辨識與追蹤
"""
import cv2
from ultralytics import YOLO
import torch
import yt_dlp
import os
import numpy as np
from collections import defaultdict, deque
from depth_integration import DepthEstimator


def get_youtube_stream_url(url):
    """取得 YouTube 影片的串流 URL"""
    print(f"正在取得串流 URL: {url}")
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info['url']
            print(f"✓ 已取得串流 URL")
            return stream_url
    except Exception as e:
        print(f"取得串流 URL 失敗: {e}")
        return None


def calculate_iou(box1, box2):
    """計算兩個邊界框的 IoU (Intersection over Union)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 計算交集區域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 計算聯集區域
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_center_distance(box1, box2):
    """計算兩個邊界框中心點的歐氏距離"""
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance


def calculate_size_similarity(box1, box2):
    """計算兩個邊界框的大小相似度"""
    width1 = box1[2] - box1[0]
    height1 = box1[3] - box1[1]
    area1 = width1 * height1
    
    width2 = box2[2] - box2[0]
    height2 = box2[3] - box2[1]
    area2 = width2 * height2
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # 計算面積比例的相似度
    ratio = min(area1, area2) / max(area1, area2)
    return ratio


def calculate_aspect_ratio(box):
    """計算邊界框的長寬比"""
    width = box[2] - box[0]
    height = box[3] - box[1]
    if height == 0:
        return 1.0
    return width / height


def calculate_movement_angle(history):
    """計算車輛移動方向角度（基於歷史位置）"""
    if len(history) < 2:
        return None
    
    # 使用最近的位置計算方向
    recent_points = list(history)[-5:]  # 最近5個點
    if len(recent_points) < 2:
        return None
    
    # 計算從最早到最晚的向量
    start_point = recent_points[0]
    end_point = recent_points[-1]
    
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    
    # 如果移動距離太小，認為沒有明確方向
    if abs(dx) < 2 and abs(dy) < 2:
        return None
    
    # 計算角度（弧度）
    angle = np.arctan2(dy, dx)
    return angle


def calculate_angle_difference(angle1, angle2):
    """計算兩個角度之間的最小差異（處理循環）"""
    if angle1 is None or angle2 is None:
        return 0
    
    diff = abs(angle1 - angle2)
    # 處理角度循環（例如 -179度 和 179度 實際很接近）
    if diff > np.pi:
        diff = 2 * np.pi - diff
    
    return diff


def create_vehicle_signature(track_data):
    """創建車輛特徵簽名用於重識別（包含尺度資訊）"""
    signature = {
        'class': track_data['class'],
        'histogram': track_data.get('histogram'),
        'avg_aspect_ratio': None,
        'last_position': None,
        'movement_direction': None,
        'last_box_size': None
    }
    
    # 計算平均長寬比
    if 'aspect_ratio_history' in track_data and len(track_data['aspect_ratio_history']) > 0:
        signature['avg_aspect_ratio'] = np.mean(list(track_data['aspect_ratio_history']))
    
    # 記錄最後位置
    if len(track_data['history']) > 0:
        signature['last_position'] = track_data['history'][-1]
    
    # 記錄移動方向
    if 'angle_history' in track_data and len(track_data['angle_history']) > 0:
        signature['movement_direction'] = track_data['angle_history'][-1]
    
    # 記錄最後的邊界框大小（用於尺度匹配）
    if 'box_size_history' in track_data and len(track_data['box_size_history']) > 0:
        signature['last_box_size'] = track_data['box_size_history'][-1]
    
    return signature


def match_with_history(detection, frame, vehicle_history, frame_count, max_distance=150):
    """將新偵測與歷史記錄匹配，返回最佳匹配的track_id或None（考慮尺度變化）"""
    best_match_id = None
    best_match_score = 0
    
    detection_hist = extract_color_histogram(frame, detection['box'])
    detection_center = detection['center']
    detection_area = (detection['box'][2] - detection['box'][0]) * (detection['box'][3] - detection['box'][1])
    
    for track_id, history_data in vehicle_history.items():
        # 檢查歷史記錄是否太舊
        if frame_count - history_data['last_seen'] > 100:  # 超過100幀不考慮
            continue
        
        features = history_data['features']
        score = 0.0
        
        # 1. 類別必須相同
        if features['class'] != detection['class']:
            continue
        
        # 2. 位置接近度（權重35%，降低以容納尺度變化）
        if features['last_position'] is not None:
            distance = np.sqrt(
                (detection_center[0] - features['last_position'][0])**2 +
                (detection_center[1] - features['last_position'][1])**2
            )
            if distance < max_distance:
                position_score = 1 - (distance / max_distance)
                score += 0.35 * position_score
            else:
                continue  # 距離太遠，跳過
        
        # 3. 顏色相似度（權重35%）
        if features['histogram'] is not None and detection_hist is not None:
            hist_similarity = compare_histograms(features['histogram'], detection_hist)
            score += 0.35 * max(0, hist_similarity)
        
        # 4. 長寬比相似度（權重20%，降低以容納角度變化）
        if features['avg_aspect_ratio'] is not None:
            detection_aspect = calculate_aspect_ratio(detection['box'])
            aspect_diff = abs(features['avg_aspect_ratio'] - detection_aspect)
            aspect_score = max(0, 1 - aspect_diff)
            score += 0.2 * aspect_score
        
        # 5. 尺度一致性（新增權重10%）- 檢查尺度變化是否合理
        if 'last_box_size' in features and features['last_box_size'] is not None:
            last_size = features['last_box_size']
            scale_ratio = detection_area / last_size if last_size > 0 else 1.0
            # 容許0.5-2.0倍的尺度變化（考慮遠離/接近）
            if 0.5 <= scale_ratio <= 2.0:
                # 變化越小分數越高
                scale_score = 1 - abs(np.log(scale_ratio)) / np.log(2)  # 使用對數尺度
                score += 0.1 * max(0, scale_score)
            else:
                score += 0  # 尺度變化過大，不給分
        
        # 更新最佳匹配
        if score > best_match_score and score > 0.5:  # 閾值0.5
            best_match_score = score
            best_match_id = track_id
    
    return best_match_id, best_match_score


def extract_color_histogram(frame, box):
    """提取邊界框內的顏色直方圖特徵（優化版）"""
    x1, y1, x2, y2 = map(int, box)
    # 確保邊界框在圖像範圍內
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # 縮小ROI以加快處理
    if roi.shape[0] > 64 or roi.shape[1] > 64:
        roi = cv2.resize(roi, (64, 64))
    
    # 計算 HSV 顏色直方圖（減少bins數量）
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [20, 24], [0, 180, 0, 256])  # 減少bins
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def compare_histograms(hist1, hist2):
    """比較兩個直方圖的相似度"""
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def create_depth_visualization(depth_map, target_width, target_height):
    """將深度圖轉成可顯示的彩色影像，並在資料無效時回傳提示圖。"""
    if depth_map is None:
        placeholder = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Depth unavailable', (20, target_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(placeholder, 'DAV2 relative depth', (20, target_height // 2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return placeholder

    try:
        depth = np.asarray(depth_map, dtype=np.float32)
        finite_mask = np.isfinite(depth)
        if not np.any(finite_mask):
            raise ValueError('depth map has no finite values')

        valid = depth[finite_mask]
        low = np.percentile(valid, 2)
        high = np.percentile(valid, 98)

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(valid))
            high = float(np.max(valid))

        if high <= low:
            norm = np.zeros_like(depth, dtype=np.uint8)
        else:
            clipped = np.clip(depth, low, high)
            norm = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)

        # 反相讓近處更亮，畫面通常更容易閱讀
        norm = 255 - norm

        depth_colormap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        depth_thumb = cv2.resize(depth_colormap, (target_width, target_height))
        cv2.rectangle(depth_thumb, (0, 0), (target_width - 1, target_height - 1), (255, 255, 255), 1)

        # 在深度窗顯示簡單數值與說明，方便辨識目前顏色對應的深度範圍
        min_val = float(np.min(valid))
        mean_val = float(np.mean(valid))
        max_val = float(np.max(valid))
        cv2.rectangle(depth_thumb, (6, 6), (target_width - 6, 74), (0, 0, 0), -1)
        cv2.rectangle(depth_thumb, (6, 6), (target_width - 6, 74), (255, 255, 255), 1)
        cv2.putText(depth_thumb, 'DAV2 relative depth', (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(depth_thumb, f'min:{min_val:.2f} mean:{mean_val:.2f}', (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(depth_thumb, f'max:{max_val:.2f} near=bright', (16, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return depth_thumb
    except Exception as e:
        fallback = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        cv2.putText(fallback, 'Depth render failed', (20, target_height // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(fallback, str(e)[:40], (20, target_height // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(fallback, 'DAV2 relative depth', (20, target_height // 2 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return fallback


class KalmanTracker:
    """卡爾曼濾波器用於預測車輛位置"""
    def __init__(self, initial_box):
        self.kf = cv2.KalmanFilter(4, 2)  # 4個狀態變數(x,y,vx,vy), 2個測量值(x,y)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # 初始化狀態
        center_x = (initial_box[0] + initial_box[2]) / 2
        center_y = (initial_box[1] + initial_box[3]) / 2
        self.kf.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        self.kf.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
    
    def predict(self):
        """預測下一個位置"""
        predicted = self.kf.predict()
        return (predicted[0][0], predicted[1][0])
    
    def update(self, measurement):
        """更新測量值"""
        self.kf.correct(np.array([[measurement[0]], [measurement[1]]], np.float32))


def calculate_matching_score(track_data, detection, frame, use_histogram=True, frame_count=0):
    """計算追蹤目標與偵測結果的匹配分數（多特徵融合，支援轉向偵測和遠離/接近）"""
    score = 0.0
    weights = {'iou': 0.35, 'distance': 0.2, 'size': 0.12, 'class': 0.15, 'color': 0.08, 'direction': 0.1}
    
    # 先計算長寬比變化，用於動態調整權重
    aspect_ratio_track = calculate_aspect_ratio(track_data['box'])
    aspect_ratio_detection = calculate_aspect_ratio(detection['box'])
    aspect_ratio_diff = abs(aspect_ratio_track - aspect_ratio_detection)
    
    # 計算尺度變化（遠離或接近鏡頭）
    track_area = (track_data['box'][2] - track_data['box'][0]) * (track_data['box'][3] - track_data['box'][1])
    detection_area = (detection['box'][2] - detection['box'][0]) * (detection['box'][3] - detection['box'][1])
    scale_ratio = detection_area / track_area if track_area > 0 else 1.0
    
    # 檢查是否正在遠離或接近鏡頭（尺度持續變化）
    is_depth_movement = False
    if 'box_size_history' in track_data and len(track_data['box_size_history']) >= 3:
        recent_sizes = list(track_data['box_size_history'])[-5:]
        # 檢查是否持續變大或變小
        size_trend = [recent_sizes[i] - recent_sizes[i-1] for i in range(1, len(recent_sizes))]
        if len(size_trend) >= 2:
            # 如果趨勢一致（都變大或都變小），則是深度移動
            positive_changes = sum(1 for x in size_trend if x > 0)
            negative_changes = sum(1 for x in size_trend if x < 0)
            if positive_changes >= len(size_trend) * 0.7 or negative_changes >= len(size_trend) * 0.7:
                is_depth_movement = True
    
    # 如果長寬比變化>10% 或 正在深度移動，調整權重
    if aspect_ratio_diff > 0.1 or is_depth_movement:
        # 計算降低比例（10%-50%變化時，IoU權重從35%逐步降到20%）
        reduction_factor = min(1.0, (aspect_ratio_diff - 0.1) / 0.4) if aspect_ratio_diff > 0.1 else 0.5
        iou_weight_reduction = 0.15 * reduction_factor  # 最多降低15%
        
        weights['iou'] = max(0.2, weights['iou'] - iou_weight_reduction)
        # 將降低的權重分配給距離和大小
        weights['distance'] += iou_weight_reduction * 0.6
        weights['size'] += iou_weight_reduction * 0.4
    
    # 1. IoU 分數（權重可能已動態調整）
    iou = calculate_iou(track_data['box'], detection['box'])
    score += weights['iou'] * iou
    
    # 2. 中心距離分數（距離越近分數越高）
    distance = calculate_center_distance(track_data['box'], detection['box'])
    # 使用預測位置計算距離
    if 'kalman' in track_data:
        predicted_pos = track_data['kalman'].predict()
        detection_center = ((detection['box'][0] + detection['box'][2]) / 2,
                          (detection['box'][1] + detection['box'][3]) / 2)
        pred_distance = np.sqrt((predicted_pos[0] - detection_center[0])**2 + 
                               (predicted_pos[1] - detection_center[1])**2)
        # 如果偵測位置接近預測位置，給予額外分數
        if pred_distance < 50:  # 50像素內
            distance = min(distance, pred_distance)
    
    max_distance = 200  # 最大可接受距離
    distance_score = max(0, 1 - (distance / max_distance))
    score += weights['distance'] * distance_score
    
    # 3. 大小相似度分數（考慮尺度變化容忍度）
    size_similarity = calculate_size_similarity(track_data['box'], detection['box'])
    # 如果正在深度移動，放寬尺度要求
    if is_depth_movement and 0.6 < scale_ratio < 1.4:  # 尺度在合理範圍內
        size_similarity = max(size_similarity, 0.7)  # 給予基本分數
    score += weights['size'] * size_similarity
    
    # 4. 類別一致性分數
    class_match = 1.0 if track_data['class'] == detection['class'] else 0.3
    score += weights['class'] * class_match
    
    # 5. 顏色直方圖相似度分數（優化：每5幀才計算一次）
    if use_histogram and 'histogram' in track_data and frame_count % 5 == 0:
        detection_hist = extract_color_histogram(frame, detection['box'])
        if detection_hist is not None:
            hist_similarity = compare_histograms(track_data['histogram'], detection_hist)
            hist_similarity = max(0, hist_similarity)  # 確保非負
            score += weights['color'] * hist_similarity
    elif 'histogram' in track_data:
        # 非計算幀給予中等分數
        score += weights['color'] * 0.5
    
    # 6. 方向一致性分數（處理轉向）
    if 'angle_history' in track_data and len(track_data['history']) >= 2:
        # 計算當前追蹤的移動方向
        track_angle = calculate_movement_angle(track_data['history'])
        
        # 計算新偵測的預期方向（使用歷史+新位置）
        temp_history = deque(track_data['history'])
        temp_history.append(detection['center'])
        detection_angle = calculate_movement_angle(temp_history)
        
        if track_angle is not None and detection_angle is not None:
            angle_diff = calculate_angle_difference(track_angle, detection_angle)
            # 角度差異越小，分數越高
            # 允許較大的角度變化（轉向時），最大容忍90度
            direction_score = max(0, 1 - (angle_diff / (np.pi / 2)))
            score += weights['direction'] * direction_score
        else:
            # 如果無法計算角度，給予中等分數
            score += weights['direction'] * 0.5
    
    # 7. 長寬比變化容忍度（車輛轉向時長寬比會改變）
    # 如果長寬比變化較大，給予額外的容忍度加分
    if aspect_ratio_diff > 0.1:  # 長寬比變化超過10%
        # 根據變化程度給予容忍度加分（補償IoU下降）
        # 變化10%時給0.05分，變化50%時給0.15分
        turning_tolerance = min(0.15, aspect_ratio_diff * 0.3)
        score += turning_tolerance
    
    return score


def is_vehicle_moving(track_data, window_size=20, threshold_2d=2.0, threshold_scale=0.015):
    """
    判斷車輛是否移動（使用 Sliding Window 方法）
    
    使用滑動視窗分析多個時間段的移動狀態，使判斷更穩定可靠。
    
    Args:
        track_data: 追蹤資料，包含 history 和 box_size_history
        window_size: 滑動視窗大小（幀數）
        threshold_2d: 2D平面移動閾值（像素/幀）
        threshold_scale: 尺度變化閾值（比例/幀）
    Returns:
        True if moving, False if stopped
    """
    position_history = track_data.get('history', deque())
    
    if len(position_history) < window_size:
        return True  # 資料不足時視為移動中
    
    # 取得最近的位置和尺度歷史
    recent_positions = list(position_history)[-window_size * 2:]  # 取兩倍視窗以便滑動
    
    # ==================== Sliding Window 分析 ====================
    
    # 儲存每個視窗的移動判斷結果
    window_results = []
    
    # 滑動視窗：每次滑動 3 幀（最平滑）
    step_size = 3
    
    for start_idx in range(0, len(recent_positions) - window_size + 1, step_size):
        window_positions = recent_positions[start_idx:start_idx + window_size]
        
        # 1. 計算此視窗內的 2D 平面移動
        window_distance = 0
        for i in range(1, len(window_positions)):
            dx = window_positions[i][0] - window_positions[i-1][0]
            dy = window_positions[i][1] - window_positions[i-1][1]
            window_distance += np.sqrt(dx**2 + dy**2)
        
        avg_distance = window_distance / (len(window_positions) - 1)
        
        # 2. 計算此視窗內的起點到終點位移
        start_pos = window_positions[0]
        end_pos = window_positions[-1]
        displacement = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        displacement_per_frame = displacement / (len(window_positions) - 1)
        
        # 判斷此視窗是否在移動
        is_window_moving_2d = (avg_distance > threshold_2d) or (displacement_per_frame > threshold_2d * 0.8)
        
        window_results.append({
            'moving_2d': is_window_moving_2d,
            'avg_distance': avg_distance,
            'displacement': displacement_per_frame
        })
    
    # ==================== 深度移動分析（Sliding Window）====================
    
    depth_window_results = []
    
    if 'box_size_history' in track_data and len(track_data['box_size_history']) >= window_size:
        recent_sizes = list(track_data['box_size_history'])[-window_size * 2:]
        
        for start_idx in range(0, len(recent_sizes) - window_size + 1, step_size):
            window_sizes = recent_sizes[start_idx:start_idx + window_size]
            
            # 計算此視窗內的尺度變化
            size_changes = []
            for i in range(1, len(window_sizes)):
                if window_sizes[i-1] > 0:
                    change = (window_sizes[i] - window_sizes[i-1]) / window_sizes[i-1]
                    size_changes.append(change)
            
            if len(size_changes) >= 3:
                # 計算平均變化率（絕對值）
                avg_size_change = np.mean([abs(c) for c in size_changes])
                
                # 計算變化的一致性（持續變大或變小）
                # 降低閾值以更靈敏地偵測遠離鏡頭（框變小）
                small_threshold = threshold_scale * 0.5  # 更低的閾值
                positive_count = sum(1 for c in size_changes if c > small_threshold)
                negative_count = sum(1 for c in size_changes if c < -small_threshold)
                consistency = max(positive_count, negative_count) / len(size_changes)
                
                # 計算總體尺度變化（起點到終點）
                total_change = (window_sizes[-1] - window_sizes[0]) / window_sizes[0] if window_sizes[0] > 0 else 0
                abs_total_change = abs(total_change)
                
                # 計算尺度變化的趨勢（正=變大/接近，負=變小/遠離）
                is_shrinking = total_change < -0.03  # 框變小 3% 以上 = 遠離
                is_growing = total_change > 0.03    # 框變大 3% 以上 = 接近
                
                # 判斷是否有深度移動（更寬鬆的條件）
                is_window_moving_depth = (
                    # 條件1: 一致性高且有明顯變化
                    (consistency >= 0.4 and avg_size_change > threshold_scale * 0.5) or
                    # 條件2: 總體變化明顯
                    (abs_total_change > 0.05) or
                    # 條件3: 明確的縮小趨勢（遠離）
                    (is_shrinking and consistency >= 0.35) or
                    # 條件4: 明確的放大趨勢（接近）
                    (is_growing and consistency >= 0.35)
                )
                
                depth_window_results.append({
                    'moving_depth': is_window_moving_depth,
                    'consistency': consistency,
                    'total_change': total_change,
                    'is_shrinking': is_shrinking,
                    'is_growing': is_growing
                })
    
    # ==================== 綜合判斷 ====================
    
    # 2D 移動判斷：超過一半的視窗顯示在移動
    if window_results:
        moving_2d_count = sum(1 for w in window_results if w['moving_2d'])
        is_moving_2d = moving_2d_count >= len(window_results) * 0.4  # 40% 視窗顯示移動
    else:
        is_moving_2d = True
    
    # 深度移動判斷：更寬鬆的條件以偵測遠離鏡頭
    is_moving_depth = False
    if depth_window_results:
        depth_moving_count = sum(1 for w in depth_window_results if w['moving_depth'])
        # 降低閾值：只要 20% 視窗顯示深度移動就算移動
        is_moving_depth = depth_moving_count >= len(depth_window_results) * 0.2
        
        # 額外檢查：如果有明確的縮小趨勢（遠離鏡頭），直接判定為移動
        shrinking_count = sum(1 for w in depth_window_results if w.get('is_shrinking', False))
        if shrinking_count >= len(depth_window_results) * 0.3:
            is_moving_depth = True
    
    # 最終判斷
    return is_moving_2d or is_moving_depth


def check_gpu():
    """檢查 GPU 可用性"""
    if torch.cuda.is_available():
        print(f"✓ GPU 可用: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA 版本: {torch.version.cuda}")
        # 顯示GPU記憶體資訊
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU 記憶體: {total_memory:.2f} GB")
        return True
    else:
        print("✗ GPU 不可用，將使用 CPU")
        return False


def detect_vehicles(video_source, model_path="yolo11l.pt", output_path="output_video.mp4", show_window=True):
    """
    使用 YOLOv11 進行車輛辨識與移動偵測（串流模式）
    
    Args:
        video_source: 影片路徑或 YouTube URL
        model_path: YOLO 模型路徑
        output_path: 輸出影片路徑（可選）
        show_window: 是否顯示即時視窗
    """
    # 檢查 GPU
    use_gpu = check_gpu()
    
    # 如果是 YouTube URL，取得串流 URL
    if video_source.startswith('http'):
        stream_url = get_youtube_stream_url(video_source)
        if stream_url is None:
            return
        video_source = stream_url
    
    # 載入 YOLO 模型
    print(f"\n載入模型: {model_path}")
    model = YOLO(model_path)
    
    # 設定使用 GPU (如果可用)
    device = 'cuda:0' if use_gpu else 'cpu'
    model.to(device)
    
    # 啟用 FP16 (Half Precision) 加速
    use_half = use_gpu
    if use_half:
        print(f"模型運行於: {device} (FP16 加速已啟用)")
    else:
        print(f"模型運行於: {device}")
    
    # 開啟影片串流
    print(f"\n正在開啟影片串流...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"無法開啟影片串流: {video_source}")
        return
    
    # 取得影片資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n影片資訊:")
    print(f"  解析度: {width}x{height}")
    print(f"  FPS: {fps}")
    if total_frames > 0:
        print(f"  總幀數: {total_frames}")

    out = None
    if output_path:
        # 先使用 mp4v（較普遍的軟編碼保證可播），若失敗再嘗試 avc1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_size = (width * 2, height)  # left: RGB, right: depth
        out = cv2.VideoWriter(output_path, fourcc, fps, out_size)
        if not out.isOpened():
            print(f"  警告: 無法使用 mp4v 編碼器，嘗試使用 avc1")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, out_size)
        print(f"  輸出檔案: {output_path}")
    
    # 車輛類別 (COCO dataset)
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    # 追蹤資料結構
    track_id_counter = 0
    tracked_vehicles = {}  # 當前追蹤的車輛
    vehicle_history = {}  # 已消失車輛的特徵記錄庫 {track_id: {'features': ..., 'last_seen': frame_count}}
    max_history_frames = 300  # 保留歷史記錄的最大幀數（約10秒）

    # 初始化深度估計器（可選）
    depth_estimator = None
    try:
        depth_estimator = DepthEstimator(encoder='vitl')
    except Exception as e:
        print(f"✗ 深度估計器初始化失敗: {e}")

    depth_map = None
    
    print(f"\n開始串流處理...")
    print("綠框 = 移動中 | 紅框 = 停止")
    print("按 'q' 鍵退出")

    if show_window:
        cv2.namedWindow('Vehicle Detection & Tracking (Press Q to quit)', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth (DAV2)', cv2.WINDOW_NORMAL)
        # 將兩個視窗設定為與影像相同大小
        try:
            cv2.resizeWindow('Vehicle Detection & Tracking (Press Q to quit)', width, height)
            cv2.resizeWindow('Depth (DAV2)', width, height)
        except Exception:
            pass
    
    frame_count = 0
    vehicle_count = 0
    moving_count = 0
    stopped_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("\n串流結束")
                break
            
            frame_count += 1

            # 週期性計算深度圖以節省效能（每5幀）
            if depth_estimator is not None and frame_count % 5 == 0:
                depth_map = depth_estimator.infer_frame(frame, input_size=518)

            # 產生深度視覺化，顯示大小與主視窗相同
            depth_vis = create_depth_visualization(depth_map, width, height)
            
            # 使用 YOLO 進行偵測（優化GPU使用）
            # imgsz: 增加圖像大小以充分利用GPU
            # half: 啟用FP16精度加速
            # verbose: 關閉詳細輸出
            # iou: 提高NMS IoU閾值以減少重複偵測
            results = model(
                frame, 
                device=device, 
                half=use_half,
                imgsz=1280,  # 增加到1280像素以提高GPU使用率
                verbose=False,
                conf=0.4,  # 信心度閾值
                iou=0.7    # 提高NMS IoU閾值從0.45到0.7，減少重複框
            )
            
            # 當前幀偵測到的車輛
            current_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # 只處理車輛類別且信心度 > 0.5
                    if cls in vehicle_classes and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 過濾太小的框（避免誤偵測或遠處噪點）
                        box_width = x2 - x1
                        box_height = y2 - y1
                        min_box_size = 80  # 最小邊長（像素）
                        if box_width < min_box_size or box_height < min_box_size:
                            continue  # 跳過太小的框

                        # 計算 bounding box 的平均深度（若可用）
                        mean_depth = None
                        if depth_map is not None and depth_estimator is not None:
                            mean_depth = depth_estimator.get_mean_depth(depth_map, [x1, y1, x2, y2])

                        current_detections.append({
                            'box': [x1, y1, x2, y2],
                            'class': cls,
                            'conf': conf,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'mean_depth': mean_depth,
                        })
            
            # 匹配偵測結果與已追蹤的車輛（改進的多特徵匹配）
            matched_tracks = set()
            matched_detections = set()
            
            # 建立匹配矩陣
            matching_scores = {}
            for track_id, track_data in list(tracked_vehicles.items()):
                for det_idx, detection in enumerate(current_detections):
                    score = calculate_matching_score(track_data, detection, frame, use_histogram=True, frame_count=frame_count)
                    matching_scores[(track_id, det_idx)] = score
            
            # 使用貪婪算法進行匹配（優先匹配分數最高的）
            sorted_matches = sorted(matching_scores.items(), key=lambda x: x[1], reverse=True)
            
            for (track_id, det_idx), score in sorted_matches:
                # 如果已經匹配過，跳過
                if track_id in matched_tracks or det_idx in matched_detections:
                    continue
                
                # 設定匹配閾值（更嚴格）
                if score > 0.4:  # 提高閾值以確保更精準的匹配
                    detection = current_detections[det_idx]
                    
                    # 更新追蹤資料
                    tracked_vehicles[track_id]['box'] = detection['box']
                    tracked_vehicles[track_id]['class'] = detection['class']
                    tracked_vehicles[track_id]['conf'] = detection['conf']
                    tracked_vehicles[track_id]['history'].append(detection['center'])
                    tracked_vehicles[track_id]['frames_since_update'] = 0
                    tracked_vehicles[track_id]['total_frames'] += 1
                    
                    # 記錄邊界框大小（用於偵測深度移動）
                    box_area = (detection['box'][2] - detection['box'][0]) * (detection['box'][3] - detection['box'][1])
                    if 'box_size_history' not in tracked_vehicles[track_id]:
                        tracked_vehicles[track_id]['box_size_history'] = deque(maxlen=20)
                    tracked_vehicles[track_id]['box_size_history'].append(box_area)
                    
                    # 更新顏色直方圖（每5幀更新一次以減少CPU負擔）
                    if frame_count % 5 == 0:
                        new_hist = extract_color_histogram(frame, detection['box'])
                        if new_hist is not None:
                            tracked_vehicles[track_id]['histogram'] = new_hist
                    
                    # 更新移動角度歷史
                    current_angle = calculate_movement_angle(tracked_vehicles[track_id]['history'])
                    if current_angle is not None:
                        if 'angle_history' not in tracked_vehicles[track_id]:
                            tracked_vehicles[track_id]['angle_history'] = deque(maxlen=10)
                        tracked_vehicles[track_id]['angle_history'].append(current_angle)
                    
                    # 更新長寬比歷史
                    current_aspect = calculate_aspect_ratio(detection['box'])
                    if 'aspect_ratio_history' not in tracked_vehicles[track_id]:
                        tracked_vehicles[track_id]['aspect_ratio_history'] = deque(maxlen=10)
                    tracked_vehicles[track_id]['aspect_ratio_history'].append(current_aspect)
                    
                    # 更新卡爾曼濾波器
                    if 'kalman' in tracked_vehicles[track_id]:
                        tracked_vehicles[track_id]['kalman'].update(detection['center'])
                    
                    matched_tracks.add(track_id)
                    matched_detections.add(det_idx)
            
            # 對未匹配的追蹤，增加未更新幀數
            for track_id in list(tracked_vehicles.keys()):
                if track_id not in matched_tracks:
                    tracked_vehicles[track_id]['frames_since_update'] += 1
            
            # 移除長時間未更新的追蹤（增加容忍度）
            for track_id in list(tracked_vehicles.keys()):
                frames_missing = tracked_vehicles[track_id]['frames_since_update']
                total_tracked = tracked_vehicles[track_id]['total_frames']
                # 如果車輛被追蹤很久，給予更長的容忍時間
                # 增加遮擋容忍：最少50幀，追蹤久的給100幀
                max_missing = 50 if total_tracked < 50 else 100
                if frames_missing > max_missing:
                    # 在刪除前，將車輛特徵加入歷史記錄
                    if total_tracked >= 10:  # 只記錄追蹤超過10幀的車輛
                        signature = create_vehicle_signature(tracked_vehicles[track_id])
                        vehicle_history[track_id] = {
                            'features': signature,
                            'last_seen': frame_count
                        }
                    del tracked_vehicles[track_id]
            
            # 為未匹配的偵測建立新追蹤（優先匹配歷史記錄）
            for det_idx, detection in enumerate(current_detections):
                if det_idx not in matched_detections:
                    # 檢查是否與現有追蹤重疊（避免重複追蹤）
                    is_duplicate = False
                    for track_id, track_data in tracked_vehicles.items():
                        iou = calculate_iou(track_data['box'], detection['box'])
                        # 如果IoU > 0.5，認為是重複的
                        if iou > 0.5:
                            is_duplicate = True
                            break
                    
                    # 如果不是重複的，嘗試匹配歷史記錄
                    if not is_duplicate:
                        matched_history_id = None
                        
                        # 先嘗試從歷史記錄中匹配（重用舊ID）
                        if len(vehicle_history) > 0:
                            matched_history_id, match_score = match_with_history(
                                detection, frame, vehicle_history, frame_count
                            )
                        
                        # 提取特徵
                        histogram = extract_color_histogram(frame, detection['box'])
                        kalman = KalmanTracker(detection['box'])
                        
                        # 計算邊界框面積
                        box_area = (detection['box'][2] - detection['box'][0]) * (detection['box'][3] - detection['box'][1])
                        
                        # 決定使用的ID
                        if matched_history_id is not None:
                            # 重用歷史ID
                            reused_id = matched_history_id
                            # 從歷史記錄中移除
                            del vehicle_history[matched_history_id]
                            print(f"  重用ID: {reused_id} (從歷史記錄恢復)")
                        else:
                            # 分配新ID
                            reused_id = track_id_counter
                            track_id_counter += 1
                        
                        tracked_vehicles[reused_id] = {
                            'box': detection['box'],
                            'class': detection['class'],
                            'conf': detection['conf'],
                            'history': deque(maxlen=30),
                            'histogram': histogram,
                            'kalman': kalman,
                            'frames_since_update': 0,
                            'total_frames': 1,
                            'angle_history': deque(maxlen=10),
                            'aspect_ratio_history': deque([calculate_aspect_ratio(detection['box'])], maxlen=10),
                            'box_size_history': deque([box_area], maxlen=20)  # 邊界框大小歷史
                        }
                        tracked_vehicles[reused_id]['history'].append(detection['center'])
            
            # 合併重複的追蹤（檢查現有追蹤之間是否重疊）
            tracks_to_remove = set()
            track_ids = list(tracked_vehicles.keys())
            for i in range(len(track_ids)):
                for j in range(i + 1, len(track_ids)):
                    track_id1 = track_ids[i]
                    track_id2 = track_ids[j]
                    
                    if track_id1 in tracks_to_remove or track_id2 in tracks_to_remove:
                        continue
                    
                    if track_id1 in tracked_vehicles and track_id2 in tracked_vehicles:
                        iou = calculate_iou(tracked_vehicles[track_id1]['box'], 
                                           tracked_vehicles[track_id2]['box'])
                        
                        # 如果兩個追蹤高度重疊（IoU > 0.6），保留追蹤時間較長的
                        if iou > 0.6:
                            if tracked_vehicles[track_id1]['total_frames'] >= tracked_vehicles[track_id2]['total_frames']:
                                tracks_to_remove.add(track_id2)
                            else:
                                tracks_to_remove.add(track_id1)
            
            # 移除重複的追蹤
            for track_id in tracks_to_remove:
                if track_id in tracked_vehicles:
                    del tracked_vehicles[track_id]
            
            # 清理過舊的歷史記錄
            history_to_remove = []
            for track_id, history_data in vehicle_history.items():
                if frame_count - history_data['last_seen'] > max_history_frames:
                    history_to_remove.append(track_id)
            for track_id in history_to_remove:
                del vehicle_history[track_id]
            
            # 繪製追蹤結果
            current_moving = 0
            current_stopped = 0
            
            for track_id, track_data in tracked_vehicles.items():
                x1, y1, x2, y2 = track_data['box']
                cls = track_data['class']
                conf = track_data['conf']
                total_tracked = track_data['total_frames']
                
                # 判斷是否移動（改進版：支援深度移動）
                is_moving = is_vehicle_moving(track_data)
                
                # 偵測是否正在轉向
                is_turning = False
                if 'aspect_ratio_history' in track_data and len(track_data['aspect_ratio_history']) >= 3:
                    recent_ratios = list(track_data['aspect_ratio_history'])[-3:]
                    ratio_variance = np.var(recent_ratios)
                    # 降低轉向偵測閾值，從0.05改為0.01（長寬比變化>10%時觸發）
                    if ratio_variance > 0.01:  # 長寬比變化較大
                        is_turning = True
                
                if is_moving:
                    color = (0, 255, 0)  # 綠色 - 移動中
                    status = "Moving"
                    current_moving += 1
                else:
                    color = (0, 0, 255)  # 紅色 - 停止
                    status = "Stopped"
                    current_stopped += 1
                
                # 追蹤品質指標（追蹤幀數越多，邊框越粗）
                thickness = 2 if total_tracked < 30 else 3
                
                # 繪製邊界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # 簡化標籤：只顯示 ID 和狀態
                label = f"ID:{track_id} {status}"
                
                # 繪製標籤
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            vehicle_count = len(tracked_vehicles)
            moving_count = current_moving
            stopped_count = current_stopped
            
            # 在畫面上顯示資訊
            info_y = 30
            # 背景框讓文字更清楚
            cv2.rectangle(frame, (5, 5), (350, 95), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (350, 95), (255, 255, 255), 2)
            
            cv2.putText(frame, f"Frame: {frame_count} | Total: {vehicle_count}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            info_y += 30
            cv2.putText(frame, f"Moving: {current_moving} | Stopped: {current_stopped}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            info_y += 30
            cv2.putText(frame, f"History Pool: {len(vehicle_history)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            
            # 顯示圖例
            legend_y = height - 65
            cv2.rectangle(frame, (5, legend_y - 10), (180, height - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, legend_y - 10), (180, height - 10), (255, 255, 255), 1)
            
            cv2.rectangle(frame, (10, legend_y), (30, legend_y + 20), (0, 255, 0), -1)
            cv2.putText(frame, "= Moving", (35, legend_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.rectangle(frame, (10, legend_y + 30), (30, legend_y + 50), (0, 0, 255), -1)
            cv2.putText(frame, "= Stopped", (35, legend_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 寫入影片幀（把偵測畫面與深度畫面左右合併輸出）
            if out is not None:
                try:
                    # 準備深度畫面：若已有在視窗上繪製的 depth_display，使用 depth_vis 作為基底並疊框
                    depth_for_out = depth_vis.copy() if depth_vis is not None else create_depth_visualization(depth_map, width, height)
                    try:
                        for track_id, track_data in tracked_vehicles.items():
                            bx = track_data.get('box')
                            if not bx:
                                continue
                            x1, y1, x2, y2 = map(int, bx)
                            is_moving = is_vehicle_moving(track_data)
                            color = (0, 255, 0) if is_moving else (0, 0, 255)
                            thickness = 2 if track_data.get('total_frames', 0) < 30 else 3
                            cv2.rectangle(depth_for_out, (x1, y1), (x2, y2), color, thickness)
                            mean_depth = None
                            try:
                                if depth_map is not None:
                                    mean_depth = DepthEstimator.get_mean_depth(depth_map, bx)
                            except Exception:
                                mean_depth = None
                            label = f"ID:{track_id}"
                            if mean_depth is not None:
                                label += f" {mean_depth:.2f}"
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(depth_for_out, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
                            cv2.putText(depth_for_out, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except Exception:
                        pass

                    # 合併左右畫面
                    left = frame
                    right = depth_for_out
                    if left.shape[:2] != right.shape[:2]:
                        right = cv2.resize(right, (left.shape[1], left.shape[0]))

                    combined = np.hstack([left, right])
                    out.write(combined)
                except Exception:
                    out.write(frame)
            
            # 顯示即時視窗
            if show_window:
                cv2.imshow('Vehicle Detection & Tracking (Press Q to quit)', frame)
                # 在深度視窗上同步繪製追蹤框與ID（若有 depth_map，顯示平均深度）
                if depth_vis is not None:
                    depth_display = depth_vis.copy()
                    try:
                        for track_id, track_data in tracked_vehicles.items():
                            bx = track_data.get('box')
                            if not bx:
                                continue
                            x1, y1, x2, y2 = map(int, bx)
                            is_moving = is_vehicle_moving(track_data)
                            color = (0, 255, 0) if is_moving else (0, 0, 255)
                            thickness = 2 if track_data.get('total_frames', 0) < 30 else 3
                            cv2.rectangle(depth_display, (x1, y1), (x2, y2), color, thickness)

                            mean_depth = None
                            try:
                                if depth_map is not None:
                                    mean_depth = DepthEstimator.get_mean_depth(depth_map, bx)
                            except Exception:
                                mean_depth = None

                            label = f"ID:{track_id}"
                            if mean_depth is not None:
                                label += f" {mean_depth:.2f}"

                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(depth_display, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
                            cv2.putText(depth_display, label, (x1 + 5, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except Exception:
                        pass

                    cv2.imshow('Depth (DAV2)', depth_display)
                
                # 按 'q' 鍵退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n使用者中斷")
                    break
            
            # 顯示進度（每秒更新一次）
            if frame_count % fps == 0:
                print(f"已處理: {frame_count} 幀 | 總車輛: {vehicle_count} | 移動: {current_moving} | 停止: {current_stopped}")
    
    except KeyboardInterrupt:
        print("\n\n偵測已中斷")
    
    finally:
        # 釋放資源
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n完成!")
        print(f"處理幀數: {frame_count}")
        print(f"追蹤車輛數: {vehicle_count}")
        print(f"最後統計 - 移動: {moving_count} | 停止: {stopped_count}")


def main():
    """主程式"""
    print("=" * 60)
    print("車輛追蹤與移動偵測系統 - YOLOv11 (增強版)")
    print("=" * 60)
    print("\n追蹤特徵：")
    print("  ✓ IoU 匹配")
    print("  ✓ 中心距離計算")
    print("  ✓ 大小相似度")
    print("  ✓ 類別一致性")
    print("  ✓ 顏色直方圖匹配")
    print("  ✓ 卡爾曼濾波器預測")
    print("  ✓ 移動方向角度追蹤")
    print("  ✓ 轉向偵測與容忍度")
    print("=" * 60)
    
    # 測試影片 URL
    video_url = "test_video/1779971075401.mp4"  # 可替換為 YouTube URL 或本地影片路徑
    
    # 執行車輛偵測與追蹤（串流模式）
    # 綠框 = 移動中的車輛
    # 紅框 = 停止的車輛
    # 簡潔顯示：只顯示 ID 和狀態
    detect_vehicles(
        video_source=video_url,
        model_path="yolo11l.pt",
        output_path="fin.mp4",  # 不保存影片
        show_window=True
    )


if __name__ == "__main__":
    main()
