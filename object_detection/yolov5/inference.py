import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5s, yolov5m, yolov5l, yolov5x, custom
img = 'https://ultralytics.com/images/zidane.jpg'  # file, Path, PIL, OpenCV, numpy, list
results = model(img)
results.show()  # .print(), .show(), .save(), .crop(), .pandas()
