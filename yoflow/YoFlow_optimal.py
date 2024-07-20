import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import csv
import scripts.src.utils as utils
#import importlib
#importlib.reload(utils)


''' 
Initialization parameters
'''
path, str_input = utils.video_selection("1c","2")
video = cv.VideoCapture(path)
cap = video
reso = 480
_, frame_1 = cap.read()
frame_1 = utils.resize_frame(frame_1, 480)

################### Optical Flow Initialization Params
OpFLOW_IDs = defaultdict(lambda : [])

hsv_mask = np.zeros_like(frame_1)
hsv_mask[:, :, 1] = 255

init_flow = True
list_mag = []; list_ang = []; list_hist = []

#############################  Yolo initialization params
yolo_pt = r'D:\yolo_models\yolov8m.pt'
model = YOLO(yolo_pt)
model.to('cuda')

track_hist = defaultdict(lambda: [])
#########################################################

########################################################
csvfile = open(r'C:\Users\Andreas Aditya\PycharmProjects\pythonProject\06_Timeseries\Optimized.csv', 'w', newline='')
csv_writer = csv.writer(csvfile)

csv_writer.writerow(['YOLO_ID','Bin_0', 'Bin_1', 'Bin_2',
                     'Bin_3', 'Bin_4', 'Bin_5', 'Bin_6', 'Bin_7', 'Frame'])
frame_count = 0
########################################################

prev_frames = {}  # Store previous frames for each track ID
gray_F2 = None

while True:
    status_cap, frame = cap.read()

    start = time.time()

    if not status_cap:
        break

    frame_count += 1

    frame = utils.resize_frame(frame, 480)

    YOLO_annot = frame.copy()

    YOLO_res = model.track(frame, persist=True)
    if YOLO_res[0].boxes.id is not None:

        YOLO_bb = YOLO_res[0].boxes.xywh
        YOLO_TrackID = YOLO_res[0].boxes.id.numpy().astype(int)

        YOLO_annot = YOLO_res[0].plot(line_width=1, labels=True, conf=0.7, probs=False)

        for box, track_id in zip(YOLO_bb, YOLO_TrackID):
            x, y, w, h = map(int, box)
            box_w = 60 / 2
            box_h = 30 / 2
            pt_1 = (int(x - box_w), int(y - box_h))
            pt_2 = (int(x + box_w), int(y + box_h))

            if track_id not in prev_frames:
                gray_F1, gray_F2, INIT_OF = utils.INIT_DenseOF(frame_1, frame)
                prev_frames[track_id] = gray_F2
            else:
                gray_F1 = prev_frames[track_id]
                gray_F2, gray_F2_new, INIT_OF = utils.AFTER_DenseOF(gray_F1, frame, INIT_OF)
                prev_frames[track_id] = gray_F2_new

            ROI_ang = utils.cut_OF(INIT_OF[:, :, 1], pt_1, pt_2)
            ROI_mag = utils.cut_OF(INIT_OF[:, :, 0], pt_1, pt_2)

            cut_mag, cut_ang = cv.cartToPolar(ROI_mag, ROI_ang, angleInDegrees=True)

            cut_ang = (360 - cut_ang) % 360
            cut_ang = cut_ang.astype(int)

            histogram_GPT = utils.hoof_GPT(cut_mag, cut_ang)
            csv_writer.writerow([track_id] + list(histogram_GPT) + [frame_count])

    end = time.time()

    fps = 1 / (end-start)

    cv.putText(YOLO_annot, f"{fps:.2f} FPS", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("YOLO-OUTPUT", YOLO_annot)
    if gray_F2 is not None:
        cv.imshow("Dense-OF", gray_F2)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

csvfile.close()
cap.release()
cv.destroyAllWindows()