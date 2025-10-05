import os
import json

base_dirs = [
    "data/UCLM_Thermal_Imaging_Dataset/Handgun",
    "data/UCLM_Thermal_Imaging_Dataset/No_Gun"
]
output_base = "data/processed/labels/all"

os.makedirs(output_base, exist_ok=True)

for base in base_dirs:
    for scenario in os.listdir(base):
        scenario_path = os.path.join(base, scenario)
        json_path = os.path.join(scenario_path, "label.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        img_w, img_h = 750, 1000  # from your JSON
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"] - 1  # YOLO: 0=Handgun, 1=Person, etc.
            x, y, w, h = ann["position"]
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            yolo_line = f"{category_id} {x_center} {y_center} {w_norm} {h_norm}\n"
            # Label filename matches image filename
            label_file = os.path.join(output_base, f"{scenario}_frame_{image_id:04d}.txt")
            with open(label_file, "a") as f:
                f.write(yolo_line)
