"""
錄製15秒原始影片 - 從YouTube串流錄製
"""
import cv2
import yt_dlp
import time


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


def record_video(video_source, output_path="recorded_15s.mp4", duration_seconds=15):
    """
    錄製指定秒數的原始影片
    
    Args:
        video_source: 影片來源（YouTube URL 或本地檔案）
        output_path: 輸出影片路徑
        duration_seconds: 錄製秒數
    """
    # 如果是 YouTube URL，取得串流 URL
    if video_source.startswith('http'):
        stream_url = get_youtube_stream_url(video_source)
        if stream_url is None:
            return
        video_source = stream_url
    
    print(f"\n正在開啟影片串流...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"無法開啟影片串流: {video_source}")
        return
    
    # 取得影片資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\n影片資訊:")
    print(f"  解析度: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  錄製時長: {duration_seconds} 秒")
    print(f"  輸出檔案: {output_path}")
    
    # 建立影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"無法建立輸出影片: {output_path}")
        cap.release()
        return
    
    print(f"\n開始錄製...")
    start_time = time.time()
    frame_count = 0
    target_frames = fps * duration_seconds
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("\n串流結束")
                break
            
            # 寫入影片
            out.write(frame)
            frame_count += 1
            
            # 顯示預覽（可選）
            cv2.imshow('Recording... (Press Q to stop)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n使用者中斷錄製")
                break
            
            # 顯示進度
            elapsed = time.time() - start_time
            if frame_count % fps == 0:
                print(f"已錄製: {elapsed:.1f}秒 / {duration_seconds}秒 ({frame_count} 幀)")
            
            # 檢查是否達到目標時長
            if frame_count >= target_frames:
                print(f"\n✓ 已完成 {duration_seconds} 秒錄製")
                break
    
    except KeyboardInterrupt:
        print("\n\n錄製已中斷")
    
    finally:
        # 釋放資源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        actual_duration = frame_count / fps if fps > 0 else 0
        print(f"\n完成!")
        print(f"錄製幀數: {frame_count}")
        print(f"實際時長: {actual_duration:.2f} 秒")
        print(f"輸出檔案: {output_path}")


def main():
    """主程式"""
    print("=" * 60)
    print("YouTube 影片錄製工具 - 15秒原始影片")
    print("=" * 60)
    
    # 測試影片 URL
    video_url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
    
    # 錄製15秒原始影片
    record_video(
        video_source=video_url,
        output_path="test_video/recorded_15s.mp4",
        duration_seconds=15
    )


if __name__ == "__main__":
    main()
