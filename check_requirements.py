# This is a sample Python script.
import cv2 as cv
import torch
import torchvision
import os
import yaml

print(os.getcwd())

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
main_path = config['main_path']['Home']

print("--path config:")
print(f"main path: {main_path}")
test_video = main_path + r'test_dataset/single_file/NT-36-56.mp4'
print(f"test vid path: {test_video}")
cap = cv.VideoCapture(test_video)
status, frame = cap.read()
if not status:
    print("path video failed")
else:
    print(f"video path: {test_video}")


print("\n--torch info:")
print(f"is cuda available? : {torch.cuda.is_available()}")
print(f"torch device: {torch.cuda.current_device()}")
print(f"torch version: {torch.__version__}")
print(f"torch vision version: {torchvision.__version__}")

print("\n--CUDA information")
print(f"CUDA enabled device: {cv.cuda.getCudaEnabledDeviceCount()}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

print("\n--OpenCV info:")
print(f"OpenCV version: {cv.__version__}")

print(f"\nOpenCV build information:")
print(cv.getBuildInformation())