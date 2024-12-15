from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model (revise the path of data.yaml on your own)
results = model.train(data='/home/joshhsieh1999/Desktop/JustinTing/banana_dataset/data.yaml', epochs=30, imgsz=640)