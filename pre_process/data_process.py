import pandas as pd
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import csv
from scripts.src import utils
import seaborn as sns
import matplotlib as plt
import dill as pickle
from scripts.PreProcess.signal_process import butter_LPF_Viz as signal_process
import os


def process_defaultdict(tracking_dictionary):
    final_df = pd.DataFrame()

    for vec_id, _ in tracking_dictionary.items():
        ## remove detection with less than
        if len(tracking_dictionary[vec_id]['Frame']) < 60:
            print(vec_id, 'removed')

        else:

            '''  process optical flow histogram '''
            ## create df for processing optical flow
            df_opflow = pd.DataFrame(tracking_dictionary[vec_id]['OF_mag'],
                                     columns=[
                                         'bins_0', 'bins_1', 'bins_2', 'bins_3',
                                         'bins_4', 'bins_5', 'bins_6', 'bins_7'])

            ## filter OF signal
            filtered_opflow = signal_process(df_opflow, cutoff=2, fs=60, order=1)
            quantiles = filtered_opflow.quantile([0.5, 0.75])
            df_transform = {}

            ## Iterate over each column to collect 0.5 & 0.75 quantiles
            for column in filtered_opflow.columns:
                df_transform[f'{column}_50'] = round(quantiles.loc[0.5, column], 0)
                df_transform[f'{column}_75'] = round(quantiles.loc[0.75, column], 0)

            ## Convert the dictionary to a DataFrame
            transformed_df = pd.DataFrame([df_transform])

            '''  process exit entry assessment '''
            df_EE = pd.DataFrame([
                {
                    'Entry': tracking_dictionary[vec_id]['Entry'],
                    'Entry_point': tracking_dictionary[vec_id]['Entry_point'],
                    'Exit': tracking_dictionary[vec_id]['Exit'],
                    'Exit_point': tracking_dictionary[vec_id]['Exit_point']
                }
            ], columns=['Entry', 'Entry_point', 'Exit', 'Exit_point'])

            '''  process data label '''
            df_label = pd.DataFrame([{
                'path_label': tracking_dictionary[vec_id]['Label'],
                'outcome': tracking_dictionary[vec_id]['is_crash?'],
            }],
                columns=['path_label', 'outcome'])

            concated_df = pd.concat([transformed_df, df_EE, df_label], axis=1)

        final_df = pd.concat([final_df, concated_df], ignore_index=True)

    return final_df

def YoFlow_main(path, yolo_models:str) -> defaultdict:

    '''
    Initialization parameters
    '''

    video = cv.VideoCapture(path)
    cap = video
    _, frame_1 = cap.read()
    frame_1 = utils.resize_frame(frame_1, 480)

    '''
    Intersection Entry Exit Parameters
    '''
    poly_1 = np.array([[118, 200], [148, 190], [155, 450], [67, 450]])  ## left lane
    poly_2 = np.array([[250, 150], [411, 130], [425, 135], [255, 160]])  ## upper lane
    poly_3 = np.array([[520, 135], [530, 130], [730, 190], [727, 200]])  ## right lane
    poly_4 = np.array([[850, 280], [850, 475], [650, 475]])  ## lower lane

    polys = [poly_1, poly_2, poly_3, poly_4]

    '''
    Yolo initialization params
    '''
    yolo_pt = yolo_models
    model = YOLO(yolo_pt)

    '''
    Tracking initialization params
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

    while True:
        status_cap, frame = cap.read()

        start = time.time()

        if not status_cap:
            break

        frame_count += 1

        frame = utils.resize_frame(frame, 480)

        YOLO_annot = frame.copy()

        ''' 
        Main Tracking Block - Order of Execution: 

            --- YOLO
            1. YOLO Detection, tracking
            2. YOLO Bounding Box localization

            --- Entry Exit
            3. Intersection Entry Exit assessment

            --- Optical Flow Block
            4. Optical Flow calculation
            5. Get ROI Optical Flow for each object
            6. Organize flows using angle

            --- Data collection & Accident labeling 
            7. organize and collect all data 
            8. manage data labeling

        '''

        ''' YOLO Block '''
        YOLO_res = model.track(frame, persist=True)
        if YOLO_res[0].boxes.id is not None:

            YOLO_bb = YOLO_res[0].boxes.xywh.cpu()
            YOLO_TrackID = YOLO_res[0].boxes.id.cpu().numpy().astype(int)

            YOLO_annot = YOLO_res[0].plot(line_width=1, labels=True, conf=0.7, probs=False)

            for box, track_id in zip(YOLO_bb, YOLO_TrackID):
                x, y, w, h = map(int, box)
                box_w = 60 / 2
                box_h = 30 / 2
                pt_1 = (int(x - box_w), int(y - box_h))  ## pt_1: left upper point
                pt_2 = (int(x + box_w), int(y + box_h))  ## pt_2: right lower point

                ''' YOLO Assessment '''
                track_hist[track_id]['Frame'].append(frame_count)

                ''' Entry Exit Assessment '''
                ## visualization of center point - comment if not needed.
                center_point = (x, y)
                cv.circle(YOLO_annot, center=center_point, radius=3, thickness=2, color=(255, 255, 255))

                newly_entered = False

                ## iterate over each polygone

                for idx, poly in enumerate(polys):
                    ## check if object center point is inside polygone
                    if utils.check_center_location(poly, center_point):
                        # Object is inside the polygon
                        if not track_hist[track_id]['Entry']:
                            # Set the first polygon index as the entry point
                            track_hist[track_id]['Entry_point'] = idx + 1
                            track_hist[track_id]['Entry'] = True
                            track_hist[track_id]['last_poly'] = idx
                        else:
                            # If a new polygon is detected and it's different from the entry polygon
                            if track_hist[track_id]['last_poly'] is not None and track_hist[track_id][
                                'last_poly'] != idx:
                                track_hist[track_id]['Exit_point'] = idx + 1
                                track_hist[track_id]['Exit'] = True
                                # Optionally reset last_poly if no further tracking is needed
                                track_hist[track_id]['last_poly'] = None

                ''' Optical flow Block '''

                ## 4. optical flow calculation
                gray_F1, gray_F2, INIT_OF = utils.Oneline_OF(frame_1, frame)

                ## 5. get ROI optical flow for each object
                ROI_ang = utils.cut_OF(INIT_OF[:, :, 1], pt_1, pt_2)
                ROI_mag = utils.cut_OF(INIT_OF[:, :, 0], pt_1, pt_2)
                cut_mag, cut_ang = cv.cartToPolar(ROI_mag, ROI_ang, angleInDegrees=True)

                ## 6. organize flows using angle
                cut_ang = (360 - cut_ang) % 360
                cut_ang = cut_ang.astype(int)

                ''' Data Collection '''

                ## 7. write collected information regarding ID
                histogram_bins = utils.HOOF_ID(cut_mag, cut_ang)
                track_hist[track_id]['OF_mag'].append(histogram_bins)

                ## 8. manage data labeling
                path_label = utils.data_labeling(path)

                if path_label.startswith('NT'):
                    track_hist[track_id]['Label'] = [path_label]
                    track_hist[track_id]['is_crash?'] = False
                else:
                    track_hist[track_id]['Label'] = [path_label]
                    track_hist[track_id]['is_crash?'] = True

        ## FPS calculation
        ## Visualization of whole process - comment if not needed
        end = time.time()
        fps = 1 / (end - start)
        frame_1 = frame

        cv.putText(YOLO_annot, f"{fps:.2f} FPS", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("YOLO-OUTPUT", YOLO_annot)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

    return track_hist