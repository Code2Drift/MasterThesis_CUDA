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


''' 
Video Configuration 
'''
path_vid = r'D:\BeamNG_dataset\Crash_8SC1\etk800-LegranSE\NT-70-40.mp4'
cap = cv.VideoCapture(path_vid)
resolution = (854, 480)


''' 
YOLO Configuration 
'''
yolo_model = r"C:\Users\Team Vaculin\PycharmProjects\test_headless_opcv\yolo_model\yolov8m.pt"
model = YOLO(yolo_model)
model.to('cuda')


'''
Entry Exit Intialization Params
'''
poly_1 = np.array([[118, 200], [148, 190], [155, 450], [67, 450]])   ## left lane
poly_2 = np.array([[250, 150], [411, 130], [425, 135], [255, 160]])  ## upper lane
poly_3 = np.array([[520, 135], [530, 130], [730, 190], [727, 200]])  ## right lane
poly_4 = np.array([[850, 280], [850, 475], [650, 475]])              ## lower lane

polys = [poly_1, poly_2, poly_3, poly_4]


'''
Tracking Information
'''
frame_count = 0
track_hist = defaultdict(lambda: {
    'Frame': [],
    'Entry': False,
    'Entry_point': None,
    'Exit': False,
    'Exit_point': None,
    'OF_mag': [],
    'is_crash?': False,
    'Label': [],
    'last_poly': None

    })


""" Initator FRAMES """
status_success, init_frame = cap.read()


if status_success:

    ########### initialization frame
    """ OPCV-YOLO Block """
    init_frame = utils.resize_frame(init_frame, 480)

    """ CUDA BLOCK """
    CUDA_frame_prev = cv.cuda_GpuMat()
    CUDA_frame_prev.upload(init_frame)
    #CUDA_frame_prev = cv.cuda.resize(CUDA_frame_prev, resolution)
    CUDA_frame_prev = cv.cuda.cvtColor(CUDA_frame_prev, cv.COLOR_BGR2GRAY)


    while True:

        ## stop process if frame did not red correctly
        start = time.time()

        """ Read First Frame """
        success, frame = cap.read()

        if not success:
            print("video ended")
            break

        current_frame = utils.resize_frame(frame, 480)


        frame_count += 1

        """ Pre-Processing """
        CUDA_frame_curr = cv.cuda_GpuMat()
        CUDA_frame_curr.upload(current_frame)
        #CUDA_frame_curr = cv.cuda.resize(CUDA_frame_curr, resolution)
        CUDA_frame_curr = cv.cuda.cvtColor(CUDA_frame_curr, cv.COLOR_BGR2GRAY)


        ''' Main YOLO Block '''
        YOLO_RESULT = model.track(frame, persist=True, conf=0.3)

        YOLO_ANNOT = frame.copy()

        if YOLO_RESULT[0].boxes.id is not None:

            ## plot box and yolo id on annotated frame
            YOLO_bb = YOLO_RESULT[0].boxes.xywh
            YOLO_trackID = YOLO_RESULT[0].boxes.id.numpy().astype(int)
            YOLO_ANNOT = YOLO_RESULT[0].plot(line_width=1, labels=False, probs=False, conf=False)

            for box, track_id in zip(YOLO_bb, YOLO_trackID):

                ''' YOLO Tracking Result '''
                x, y, w, h = map(int, box)
                box_w = 60 / 2
                box_h = 30 / 2
                pt_1 = (int(x - box_w), int(y - box_h))  ## pt_1: left upper point
                pt_2 = (int(x + box_w), int(y + box_h))  ## pt_2: right lower point

                track_hist[track_id]['Frame'].append(frame_count)

                center_point = (x, y)
                cv.circle(YOLO_ANNOT, center=center_point, radius=3, thickness=2, color=(255, 255, 255))

                ''' Intersection Entry Exit Assessment '''
                for idx, poly in enumerate(polys):
                    ## check if object center point is inside polygone
                    if utils.check_center_location(poly, center_point):
                        if not track_hist[track_id]['Entry']:
                            ## Set the first polygon index as the entry point
                            track_hist[track_id]['Entry_point'] = idx + 1
                            track_hist[track_id]['Entry'] = True
                            track_hist[track_id]['last_poly'] = idx
                        else:
                            ## If a new polygon is detected and it's different from the entry polygon
                            if track_hist[track_id]['last_poly'] is not None and track_hist[track_id][
                                'last_poly'] != idx:
                                track_hist[track_id]['Exit_point'] = idx + 1
                                track_hist[track_id]['Exit'] = True
                                # Optionally reset last_poly if no further tracking is needed
                                track_hist[track_id]['last_poly'] = None

                """ Optical Flow """

                """
                Pipeline - 1, note:
                1. OF calculation is successfull, can run the whole video
                2. SIgnificant calculation error on flow magnitude. Magnitude between Cut_OF and uncutted OF is the same.
                    meaning the pointer for bbox referene point is faulty. 

                3. 
                """
                # create optical flow instance
                cuda_flow = cv.cuda_FarnebackOpticalFlow.create(
                    5,          # num levels
                    0.5,        # pyramid scale
                    False,      # Fast pyramid
                    13,         # winSize
                    10,         # numIters
                    5,          # polyN
                    1.1,        # PolySigma
                    0,          # flags
                )


                # calculate optical flow
                cuda_flow = cv.cuda_FarnebackOpticalFlow.calc(
                    cuda_flow, CUDA_frame_prev, CUDA_frame_curr, None,
                )

                cuda_flow_x = cv.cuda_GpuMat(cuda_flow.size(), cv.CV_32FC1)
                cuda_flow_y = cv.cuda_GpuMat(cuda_flow.size(), cv.CV_32FC1)
                cv.cuda.split(cuda_flow, [cuda_flow_x, cuda_flow_y])

                cuda_mag, cuda_ang = cv.cuda.cartToPolar(
                    cuda_flow_x, cuda_flow_y, angleInDegrees=True
                )

                angle = cuda_ang.download()
                mag_flow = cuda_mag.download()

                ROI_ang = utils.cut_ndarray(angle, pt_1, pt_2)
                ROI_mag = utils.cut_ndarray(mag_flow, pt_1, pt_2)

                """
                Pipeline - 1, DIrectly cutting OF_mag in Cuda_GpuMat():
                1. use function: cut_OF_cuda
                """
                # cuda_flow = cv.cuda_FarnebackOpticalFlow.calc(
                #     cuda_flow, CUDA_frame_prev, CUDA_frame_curr, None
                #
                # )
                #
                # cuda_flow_x = cv.cuda_GpuMat(cuda_flow.size(), cv.CV_32FC1)
                # cuda_flow_y = cv.cuda_GpuMat(cuda_flow.size(), cv.CV_32FC1)
                # cv.cuda.split(cuda_flow, [cuda_flow_x, cuda_flow_y])
                #
                # cuda_mag, cuda_ang = cv.cuda.cartToPolar(
                #     cuda_flow_x, cuda_flow_y, angleInDegrees=True
                # )
                #
                # ANG = utils.cut_OF_cuda(cuda_ang, pt_1, pt_2)
                # MAG = utils.cut_OF_cuda(cuda_mag, pt_1, pt_2)
                #
                # ROI_ang = ANG.download()
                # ROI_mag = MAG.download()






        CUDA_frame_prev = CUDA_frame_curr

        end = time.time()
        fps = 1 / (end - start)

        # Frame-by-Frame plot of image
        cv.putText(YOLO_ANNOT, f"{fps:.2f} FPS", (440, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Cuda Frame", YOLO_ANNOT)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
cap.release()

print("ROI magnitude:")
print(f"maximum flow: {ROI_mag}")
print(f"minimum flow: {ROI_mag}")

# print(f"\nWhole Frame Magnitude")
# print(f"maximum flow: {mag_flow.max()}")
# print(f"minimum flow: {mag_flow.min()}")

print("\nROI Angle")
print(f"maximum angle: {ROI_ang.max()}")
print(f"minimum angle: {ROI_ang.min()}")

# print(f"\nwhole Frame Angle")
# print(f"maximum angle: {angle.max()}")
# print(f"minimum angle: {angle.min()}")

# print(f"\nFUll Flow: {full_flow}")
# print(track_hist)