import cv2
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 替换摄像头为本地视频
video_path = r"*****************"  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频播放结束时跳出循环

    # 使用 YOLO 进行目标检测
    results = model(frame)

    # 显示带有检测框的视频帧
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Detection', annotated_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
