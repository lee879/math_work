import cv2

# 输入视频文件路径
input_video_path = r'D:\pj\math\math_work\data\video\Fog20200313000026.mp4'  # 替换为你的视频文件路径

# 输出图像文件保存路径
output_image_path = r'D:\pj\math\math_work\video_frames/'  # 替换为你想要保存图像的文件夹路径

# 打开输入视频文件
cap = cv2.VideoCapture(input_video_path)

# 获取视频的帧率
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 计算要提取的帧间隔，这里设置为每隔100帧提取一帧
frame_interval = 375
begian_interval = -2350

begian_interval_1 = 68351 + begian_interval #46分钟的n的帧率
begian_interval_2 = 70000 + begian_interval - 1275 #+ 1350 #- 2775#1小时02分的帧率 - 2775

# 初始化帧计数器
frame_count = begian_interval
batch = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break  # 退出循环，视频结束
    if (frame_count > begian_interval_1) & (frame_count <= begian_interval_2):
        frame_count += 1
        if frame_count == begian_interval_2:
            frame_count = 90000
        print(frame_count)
        continue
    else:
        # 如果帧计数能够被frame_interval整除，保存这一帧图像
        if (frame_count>=0) & (frame_count % frame_interval == 0) :
            # 生成图像文件名，使用帧计数作为文件名
            image_file = output_image_path + f"{int(frame_count/frame_interval)}.jpg"
            # 保存帧图像
            cv2.imwrite(image_file, frame)
            batch += 1
            print(batch)
    frame_count += 1

# 释放VideoCapture对象
cap.release()
# 关闭所有窗口
cv2.destroyAllWindows()