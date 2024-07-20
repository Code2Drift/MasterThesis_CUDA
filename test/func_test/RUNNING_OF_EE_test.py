from ultralytics import YOLO
import cv2 as cv
import numpy as np
from collections import defaultdict
import time
from src import utils



"""" Testing POly gone """
poly_1 = np.array([[118, 200], [148, 190], [155, 450], [67, 450]])
poly_2 = np.array([[250,150], [411, 130], [425, 135], [255, 160]])
poly_3 = np.array([[520, 135], [530, 130], [730,190],[727, 200]])
poly_4 = np.array([[850, 280], [850, 475], [650, 475]])

polys = [poly_1, poly_2, poly_3, poly_4]





'''' define tracking interest for each vehicle'''
id_info = defaultdict(lambda: {
    # 'last_point': [],
    # 'frame_collected': 0,
    'ever_inside_poly': False,
    'polygone_entered': [],
    'OF_calc': {
        'magnitude' : [],
        'angle'     : [],
        #'histogram' : []
    }

})

gray_F2 = None


### Define yolo model to be used
model = YOLO('yolo_models/yolov8m.pt')
model.to('cuda')


'''define codec'''
#vid_path = r"D:\BeamNG_dataset\Crash_8\etk800-Marshall\55-62.mp4"
vid_path = r"D:\BeamNG_dataset\Crash_8SC1\etk800-Marshall\44-58.mp4"
cap = cv.VideoCapture(vid_path)


'''Define intialization for Optical Flow'''

## read the first frame for initialization in first OF
_, frame_1 = cap.read()
frame_1 = utils.resize_frame(frame_1, 480)
init_flow = True

# create a masking for display gray frame
hsv_mask = np.zeros_like(frame_1)
hsv_mask[:, :, 1] = 255




''' Main video loop'''
while cap.isOpened():
    ### set up initialization parameter
    success, frame = cap.read()

    ## break if video frame is not read correctly
    if not success:
        break

    ## start time and resize frame to 480
    start = time.time()
    frame = utils.resize_frame(frame, 480)

    ## initate yolo tracking
    annot_frame = frame.copy()
    result = model.track(frame, persist=True, save=False, tracker='bytetrack.yaml', conf=0.3)
    if result[0].boxes.id is not None:

        boxes = result[0].boxes.xywh
        track_id = result[0].boxes.id.numpy().astype(int)
        annot_frame = result[0].plot(line_width=1, labels=True, probs=False, conf=False)

        for box, track_id in zip(boxes, track_id):
            x, y, w, h = box


            """Block - Optical FLow Calculation"""
            box_w = 80 / 2
            box_h = 60 / 2
            pt_1 = (int(x - box_w), int(y - box_h))
            pt_2 = (int(x + box_w), int(y + box_h))
            cv.rectangle(annot_frame,pt1=pt_1, pt2=pt_2, color=(255,255,255), thickness=1)

            if 'gray_F1' not in id_info[track_id]:
                gray_F1, gray_F2, INIT_OF = utils.INIT_DenseOF(frame_1, frame)
                id_info[track_id]['gray_F1'] = gray_F1
                id_info[track_id]['INIT_OF'] = INIT_OF

            else:
                gray_F1 = id_info[track_id]['gray_F1']
                INIT_OF = id_info[track_id]['INIT_OF']

                gray_F2, gray_F2_new, INIT_OF = utils.AFTER_DenseOF(gray_F1, frame, INIT_OF)
                id_info[track_id]['gray_F1'] = gray_F2_new
                id_info[track_id]['INIT_OF'] = INIT_OF

            ROI_ang = utils.cut_OF(INIT_OF[:, :, 1], pt_1, pt_2)
            ROI_mag = utils.cut_OF(INIT_OF[:, :, 0], pt_1, pt_2)

            cut_mag, cut_ang = cv.cartToPolar(ROI_mag, ROI_ang, angleInDegrees=True)

            cut_ang = (360 - cut_ang) % 360
            cut_ang = cut_ang.astype(int)

            histogram_GPT = utils.hoof_GPT(cut_mag, cut_ang)

            ## Store optical flow information
            id_info[track_id]['OF_calc']['magnitude'].append(cut_mag)
            id_info[track_id]['OF_calc']['angle'].append(cut_ang)





            """Block - Main Exit and Entry Assessment"""
            center_point = (int(x), int(y))
            cv.circle(annot_frame, center=center_point, radius=3, thickness=2, color=(255,255,255))
            # id_info[track_id]['last_point'] = center_point
            # id_info[track_id]['frame_collected'] += 1

            newly_entered = False

            for idx, poly in enumerate(polys):
                if utils.check_center_location(poly, center_point):
                    if (idx + 1) not in id_info[track_id]['polygone_entered']:
                        id_info[track_id]['ever_inside_poly'] = True
                        id_info[track_id]['polygone_entered'].append(idx + 1)
                        newly_entered = True


            if not newly_entered and id_info[track_id]['ever_inside_poly']:
                pass


    annot_frame = utils.draw_lane_area(annot_frame)

    ## end time and calculate fps
    end = time.time()
    fps = 1 / (end-start)




    '''  Print Text or Line on annotated image '''
    cv.putText(annot_frame, f"{fps:.2f} FPS", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    start_x, start_y = 20, 50
    line_height = 25

    for track_id, info in sorted(id_info.items()):
        current_y = start_y + (track_id - 1) * (line_height * 2)  # Adjust for each track ID

        #ever_inside_text = f"ID {track_id} Ever Inside Poly? {'Yes' if info['ever_inside_poly'] else 'No'} - frame collected: {info['frame_collected']}"
        ever_inside_text = f"ID {track_id} Ever Inside Poly? {'Yes' if info['ever_inside_poly'] else 'No'}"
        cv.putText(annot_frame, ever_inside_text, (start_x, current_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)

        poly_entered_text = f"Entered Polys: {info['polygone_entered']}"
        cv.putText(annot_frame, poly_entered_text, (start_x, current_y + line_height), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)





    """Draw area of exit start and annotation of area"""
    # ## line left
    # cv.putText(annot_frame, "lane-1", (100, 180), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #
    # ## line top
    # cv.putText(annot_frame, "lane-2", (400, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    # ## line below
    # cv.putText(annot_frame, "lane-3", (720, 230), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #
    # ## line right
    # cv.putText(annot_frame, "lane-4", (550, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    #
    # # out.write(annot_frame)
    # cv.imshow('testing', annot_frame)




    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv.destroyAllWindows()

print(id_info)
