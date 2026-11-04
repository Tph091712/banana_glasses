from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model.
# DATA_YAML: path to the data.yaml of your dataset (e.g. exported from Roboflow); defaults to ./data.yaml
import os
DATA_YAML = os.environ.get("DATA_YAML", "data.yaml")
results = model.train(data=DATA_YAML, epochs=30, imgsz=640)
