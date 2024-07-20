import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import csv
from src import utils



path_vid = r'D:\BeamNG_dataset\Crash_8SC1\etk800-LegranSE\NT-70-40.mp4'
cap = cv.VideoCapture(path_vid)



""" Define Initator Variables """



#### read first frame
status_success, prev_frame = cap.read()

if status_success:

    ########### additional cuda transfer for BGR previous frame
    prev_frame = utils.resize_frame(prev_frame, 480)


    while True:

        """ Read First Frame """
        ## start time for complete process and frame reading
        start_fulltime = time.time()

        ret, frame = cap.read()

        ## stop processing if ran out of frames
        if not ret:
            break

        """ Pre-Processing """
        frame = utils.resize_frame(frame, 480)

        """ Optical Flow """
        start_OF = time.time()
        gray_f1, gray_f2, flow = utils.Oneline_OF(frame, prev_frame)

        mag_flow , ang_flow = cv.cartToPolar(
            flow[:, :, 0], flow[:, :, 1], angleInDegrees=True
        )


        prev_frame = frame

        ## TOCK complete processing time
        end_complete = time.time()
        fps = 1 / (end_complete - start_fulltime)

        # Frame-by-Frame plot of image
        cv.putText(frame, f"{fps:.2f} FPS", (440, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Cuda Frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
cap.release()


print("mag_flow")
print(f"maximum flow: {mag_flow.max()}")
print(f"minimum flow: {mag_flow.min()}")

print("\nmax angle")
print(f"maximum angle: {ang_flow.max()}")
print(f"minimum angle: {ang_flow.min()}")