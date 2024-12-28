import cv2
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model


# 手动创建模型架构 (你可以根据自己的需求修改这里的模型)
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # 二分类输出
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 加载权重（手动加载权重的步骤）
def load_custom_weights(model, model_path):
    try:
        model.load_weights(model_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")


# 你的模型路径
model_path = r"E:\下载\fer2013_big_XCEPTION.54-0.66.hdf5"

# 创建模型
model = create_model()

# 加载权重
load_custom_weights(model, model_path)

# 加载Haar人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 启动摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image!")
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 处理每一张检测到的人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 提取人脸区域并调整大小为模型所需尺寸
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (64, 64))

        # 归一化并进行预测
        face_resized = np.expand_dims(face_resized, axis=0)  # 增加批次维度
        face_resized = face_resized.astype('float32') / 255.0  # 归一化

        predictions = model.predict(face_resized)

        # 如果预测值大于0.5，认为是男性，否则为女性
        gender = 'Male' if predictions[0][0] > 0.5 else 'Female'

        # 在图像上显示性别
        cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2, cv2.LINE_AA)

    # 显示结果
    cv2.imshow('Gender Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

