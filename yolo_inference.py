from ultralytics import YOLO
import torch

model = YOLO('models/best.pt')

results = model.predict("input_data/08fd33_4.mp4", save=True)
print(results)
print("========================================")
for box in results[0].boxes:
    print(box)