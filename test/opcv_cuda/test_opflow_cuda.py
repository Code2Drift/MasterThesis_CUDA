import cv2 as cv
import numpy as np
from src import utils
import time
import pandas as pd


path_vid = r'D:\BeamNG_dataset\Crash_8SC1\etk800-LegranSE\NT-70-40.mp4'
cap = cv.VideoCapture(path_vid)
fps = cap.get(cv.CAP_PROP_FPS)
num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)


""" Define Initator Variables """

time_readframe = []
time_complete_process = []
time_preprocess = []
time_OpticalFlow = []
fps_collect = []
init_flow = True

#### read first frame
status_success, prev_frame = cap.read()

if status_success:

    ########### additional cuda transfer for BGR previous frame
    frame = utils.resize_frame(prev_frame, 480)
    gpu_frame = cv.cuda_GpuMat()
    gpu_frame.upload(frame)

    ########### additional cuda transfer for previous frame
    prev_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gpu_prev = cv.cuda_GpuMat()
    gpu_prev.upload(prev_frame)

    while True:

        """ Read First Frame """
        ## start time for complete process and frame reading
        start_fulltime = time.time()
        start_readtime = time.time()

        ret, frame = cap.read()
        gpu_frame.upload(frame)

        ## tock read time
        end_readtime = time.time()
        time_readframe.append(round(end_readtime - start_readtime, 5))

        ## stop processing if ran out of frames
        if not ret:
            break

        """ Pre-Processing """
        ## TICK - for preprocessing image, resize and BGR2GRAY
        start_preprocess = time.time()
        gpu_frame = cv.cuda.resize(gpu_frame, (854, 480))
        gpu_current = cv.cuda.cvtColor(gpu_frame, cv.COLOR_BGR2GRAY)

        ## TOCK - for preprocessing
        end_preprocess = time.time()
        time_preprocess.append(round(end_preprocess - start_preprocess, 5))

        """ Optical Flow """
        start_OF = time.time()

        # create optical flow instance
        gpu_flow = cv.cuda_FarnebackOpticalFlow.create(
            5, 0.5, False, 15, 3, 5, 1.2, 0,
        )
        # calculate optical flow
        gpu_flow = cv.cuda_FarnebackOpticalFlow.calc(
            gpu_flow, gpu_prev, gpu_current, None,
        )

        end_OF = time.time()
        time_OpticalFlow.append(round(end_OF - start_OF, 5))

        gpu_prev = gpu_current

        ## TOCK complete processing time
        end_complete = time.time()
        fps = 1 / (end_complete - start_fulltime)
        time_complete_process.append(round(end_complete - start_fulltime, 5))
        fps_collect.append(round(fps, 2))

        ## download uploaded images
        frame = gpu_frame.download()
        frame2 = gpu_current.download()

        # Frame-by-Frame plot of image
        cv.putText(frame2, f"{fps:.2f} FPS", (440, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Cuda Frame", frame2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
cap.release()

# Create DataFrame
df_dict = {
    "complete_time":time_complete_process,
    "fps": fps_collect,
    "preprocess": time_preprocess,
    "OF_processing": time_OpticalFlow,
    "read_frame":time_readframe[1:]
}

for key, value in df_dict.items():
    print(f"{key} has {len(value)} items inside")
    print(" ")