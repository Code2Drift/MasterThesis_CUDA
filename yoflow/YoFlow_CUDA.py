import pandas as pd
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import csv
from src import utils
import seaborn as sns
import matplotlib as plt
import os

path_vid = r'D:\BeamNG_dataset\Crash_8SC1\etk800-LegranSE\NT-70-40.mp4'
cap = cv.VideoCapture(path_vid)
resolution = 480


""" Define Initator Variables """


status_success, prev_frame = cap.read()

if status_success:

    ########### initialization frame
    frame = utils.resize_frame(prev_frame, resolution)
    gpu_frame = cv.cuda_GpuMat()
    gpu_frame.upload(frame)

    ########### additional cuda transfer for previous frame
    prev_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gpu_prev = cv.cuda_GpuMat()
    gpu_prev.upload(prev_frame)

    while True:

        """ Read First Frame """
        ret, frame = cap.read()
        gpu_frame.upload(frame)

        ## stop process if frame did not red correctly
        start = time.time()

        if not ret:
            break

        """ Pre-Processing """
        gpu_frame = cv.cuda.resize(gpu_frame, (854, 480))
        gpu_current = cv.cuda.cvtColor(gpu_frame, cv.COLOR_BGR2GRAY)



        """ Optical Flow """

        # create optical flow instance
        gpu_flow = cv.cuda_FarnebackOpticalFlow.create(
            5, 0.5, False, 15, 10, 5, 1.1, 0,
        )
        # calculate optical flow
        gpu_flow = cv.cuda_FarnebackOpticalFlow.calc(
            gpu_flow, gpu_prev, gpu_current, None,
        )

        gpu_flow_x = cv.cuda_GpuMat(gpu_flow.size(), cv.CV_32FC1)
        gpu_flow_y = cv.cuda_GpuMat(gpu_flow.size(), cv.CV_32FC1)
        cv.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

        gpu_mag, gpu_ang = cv.cuda.cartToPolar(
            gpu_flow_x, gpu_flow_y, angleInDegrees=True
        )

        angle = gpu_ang.download()
        mag_flow = gpu_mag.download()

        ## Orgnaize flow based on their angle


        ## download uploaded images
        frame = gpu_frame.download()
        frame2 = gpu_current.download()

        end = time.time()
        fps = 1 / (end - start)
        # Frame-by-Frame plot of image
        cv.putText(frame2, f"{fps:.2f} FPS", (440, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Cuda Frame", frame2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
cap.release()

print("mag_flow")
print(f"maximum flow: {mag_flow.max()}")
print(f"minimum flow: {mag_flow.min()}")

print("\nmax angle")
print(f"maximum flow: {angle.max()}")
print(f"minimum flow: {angle.min()}")
