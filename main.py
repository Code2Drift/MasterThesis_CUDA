# This is a sample Python script.
import cv2 as cv
import torch
import torchvision


print("\n--torch info:")
print(f"torch device: {torch.cuda.current_device()}")
print(f"torch version: {torch.__version__}")
print(f"torch vision version: {torchvision.__version__}")


print("\n--OpenCV info:")
print(f"OpenCV version: {cv.__version__}")
print("CUDA information")
print(f"CUDA enabled device: {cv.cuda.getCudaEnabledDeviceCount()}")
print(f"\nOpenCV build information:")
print(cv.getBuildInformation())

