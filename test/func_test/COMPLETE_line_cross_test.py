import pandas as pd
import torch
from ultralytics import YOLO
import cv2 as cv
import numpy as np
from collections import defaultdict
import time
from scripts.src import utils



vid_path = r'E:\01_Programming\Py\Masterarbeit_BeamNG\test_data\NT-34-34.mp4'

model = YOLO('yolo_models/yolov8m.pt')
model.to('cuda')
cap = cv.VideoCapture(vid_path)

'''' define tracking method for each vehicle'''
track_hist = defaultdict(lambda: [])

## keep track of object that have crossed the line
crossed_objects = {}

while cap.isOpened():
    success, frame = cap.read()

    ## break if video frame is not read correctly
    if not success:
        break

    ## start time and resize frame to 480
    start = time.time()
    frame = utils.resize_frame(frame, 480)

    ## initate yolo tracking
    annot_frame = frame.copy()
    result = model.track(frame, persist=True, save=False, tracker='bytetrack.yaml', conf=0.7)
    if result[0].boxes.id is not None:

        boxes = result[0].boxes.xywh
        track_id = result[0].boxes.id.numpy().astype(int)
        annot_frame = result[0].plot(line_width=1, labels=True, probs=False)

        for box, track_id in zip(boxes, track_id):
            x, y, w, h = box
            track = track_hist[track_id]
            track.append((float(x), float(y)))

            # tracking_data = tracking_data._append({
            #     "frame": cap.get(cv.CAP_PROP_POS_FRAMES),
            #     "track_id": track_id,
            #     "x": x,
            #     "y": y
            # }, ignore_index=True)

    ## end time and calculate fps
    end = time.time()
    fps = 1 / (end-start)

    '''  Print Text or Line on annotated image '''
    annot_frame = utils.draw_lane_area(annot_frame)


    cv.imshow('testing', annot_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


# print(tracking_data)
# tracking_data.to_csv(r'E:\01_Programming\Py\Masterarbeit_BeamNG\scripts\testing\test_tracking_coor\test_tracking.csv', index=False)