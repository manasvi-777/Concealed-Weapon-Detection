import os
import cv2

base_dirs = [
    "data/UCLM_Thermal_Imaging_Dataset/Handgun",
    "data/UCLM_Thermal_Imaging_Dataset/No_Gun"
]
output_base = "data/processed/images/all"

os.makedirs(output_base, exist_ok=True)

for base in base_dirs:
    for scenario in os.listdir(base):
        scenario_path = os.path.join(base, scenario)
        video_path = os.path.join(scenario_path, "video.mp4")
        if not os.path.exists(video_path):
            continue
        cap = cv2.VideoCapture(video_path)
        idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Unique filename: scenario_frame_xxxx.jpg
            filename = f"{scenario}_frame_{idx:04d}.jpg"
            cv2.imwrite(os.path.join(output_base, filename), frame)
            idx += 1
        cap.release()
