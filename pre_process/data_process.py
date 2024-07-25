import cv2
import cv2 as cv
from collections import defaultdict
import time
import pandas as pd
import numpy as np
import dill as pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pre_process.signal_process import  butter_LPF_Viz as signal_process
from src import utils
from pathlib import Path
import os
import yaml
from ultralytics import YOLO



"""
Load necessary object
"""
main_path = Path(__file__).parent.parent.absolute()
config_path = os.path.join(main_path, 'config.yaml')

### Path object
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

## Different Objects
polys = utils.load_EE_params()
track_hist = utils.load_defaultdict()


### YOLO configuration
yolo_models = os.path.join(main_path, config['YOLO']['yolo8_m'])
model = YOLO(yolo_models)
model.to('cuda')
resolution = (854, 480)


def YOFLOW_main(video_path, target_name):

    '''
    Tracking Information
    '''
    frame_count = 0
    track_hist = utils.load_defaultdict()
    fps_list = []

    cap = cv2.VideoCapture(video_path)

    """ Initator FRAMES """
    status_success, init_frame = cap.read()

    if status_success is not True:
        print("fail to read frame")

    else:
            ########### initialization frame

            """ OPCV-YOLO Block """
            init_frame = utils.resize_frame(init_frame, 480)
            init_frame = utils.cut_frame(init_frame)

            """ CUDA BLOCK """
            CUDA_frame_prev = utils.send2cuda(init_frame)


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
                        7. iterate until vehicle is not detected or tracking failed
                        8. manage data labeling

            '''


            while True:

                ## stop process if frame did not red correctly
                """ Read First Frame """
                success, current_frame = cap.read()

                ## start time to calculate FPS
                start = time.time()

                if not success:
                    print("video ended")
                    break

                """ OPCV Block for second frame """
                current_frame = utils.resize_frame(current_frame, 480)
                current_frame = utils.cut_frame(current_frame)
                frame_count += 1

                """ Pre-Processing """
                CUDA_frame_curr = utils.send2cuda(current_frame)

                ''' Main YO_FLOW Block '''
                YOLO_RESULT = model.track(current_frame, persist=True, conf=0.3)
                YOLO_ANNOT = current_frame.copy()

                if YOLO_RESULT[0].boxes.id is not None:
                    YOLO_bb = YOLO_RESULT[0].boxes.xywh
                    YOLO_trackID = YOLO_RESULT[0].boxes.id.numpy().astype(int)

                    ## for visualization purpose
                    YOLO_ANNOT = YOLO_RESULT[0].plot(line_width=1, labels=False, probs=False, conf=False)

                    """ Main Logic for YOFLOW """
                    for box, track_id in zip(YOLO_bb, YOLO_trackID):

                        ''' YOLO Tracking Result '''
                        x, y, w, h = map(int, box)
                        box_w = 60 / 2
                        box_h = 30 / 2
                        pt_1 = (int(x - box_w), int(y - box_h))  ## pt_1: left upper point
                        pt_2 = (int(x + box_w), int(y + box_h))  ## pt_2: right lower point
                        track_hist[track_id]['Frame'].append(frame_count)
                        center_point = (x, y)

                        # for visualization purpose
                        cv.circle(YOLO_ANNOT, center=center_point, radius=3, thickness=2, color=(255, 255, 255))


                        ''' Intersection Entry Exit Assessment '''

                        ### Ver. 2 EE-Assessment
                        utils.EE_Assessment(polys, center_point, default_dict=track_hist,
                                            vehicle_id=track_id)

                        ''' Optical Flow CUDA Calculation'''

                        ROI_mag, ROI_ang = utils.cuda_opflow(
                            cuda_prev = CUDA_frame_prev,
                            cuda_current = CUDA_frame_curr,
                            pt_1 = pt_1,
                            pt_2 = pt_2
                        )

                        #### 1. Choice : Sum of Flow in Clockwise fashion
                        histogram_bins = utils.HOOF_sum_experimental(ROI_mag, ROI_ang)

                        #### 2. Choice : Median of Flow in Clockwise fashion
                        # histogram_bins = utils.HOOF_median(ROI_mag, ROI_ang)

                        #### 3. Choice : Average of Flow in Clockwise fashion
                        # histogram_bins = utils.HOOF_avg(ROI_mag, ROI_ang)

                        """ Add Calculated flow to each detected object """
                        track_hist[track_id]['OF_mag'].append(histogram_bins)


                CUDA_frame_prev = CUDA_frame_curr

                end = time.time()
                fps = 1 / (end - start)

                fps_list.append(fps)

                # visualization Frame-by-Frame plot of image
                cv.putText(YOLO_ANNOT, f"{fps:.2f} FPS", (440, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow(target_name, YOLO_ANNOT)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

    cv.destroyAllWindows()
    cap.release()

    """
    Data labeling 
    """
    path_label = utils.data_labeling(video_path)
    crash_status = not path_label.startswith('NT')
    print(" ")
    print(" ")
    print(" ")

    print(path_label)

    print(" ")
    print(" ")
    print(" ")


    for track_id in track_hist.keys():
        track_hist[track_id]['label'] = path_label
        track_hist[track_id]['is_crash?'] = crash_status

    return track_hist


def process_defaultdict(tracking_dictionary):
    final_df = pd.DataFrame()

    for vehicle_id, _ in tracking_dictionary.items():

        if len(tracking_dictionary[vehicle_id]['Frame']) < 120:
            print(vehicle_id, 'removed')
            concated_df = pd.DataFrame()

        else:
            ''' Process Optical Flow Histogramm '''
            ## convert OF_mag to optical flow bins
            df_opflow2 = pd.DataFrame(tracking_dictionary[vehicle_id]['OF_mag'],
                                      columns=[
                                          'bins_0', 'bins_1', 'bins_2', 'bins_3',
                                          'bins_4', 'bins_5', 'bins_6', 'bins_7'])
            ## process signal filtering
            filtered_opflow = signal_process(df_opflow2, cutoff=2, fs=60, order=1)
            quantiles = filtered_opflow.quantile([0.75])
            df_transform = {}

            ## Iterate over each column to collect 0.5 & 0.75 quantiles
            for column in filtered_opflow.columns:
                # df_transform[f'{column}_50'] = round(quantiles.loc[0.5, column], 0)
                df_transform[f'{column}_75'] = round(quantiles.loc[0.75, column], 0)

            ## Convert the dictionary to a DataFrame
            transformed_df = pd.DataFrame([df_transform])

            '''  process exit entry assessment '''
            df_EE = pd.DataFrame([
                {
                    'Entry': tracking_dictionary[vehicle_id]['Entry'],
                    'Entry_point': tracking_dictionary[vehicle_id]['Entry_point'],
                    'Exit': tracking_dictionary[vehicle_id]['Exit'],
                    'Exit_point': tracking_dictionary[vehicle_id]['Exit_point']
                }
            ], columns=['Entry', 'Entry_point', 'Exit', 'Exit_point'])

            '''  process data label '''
            df_label = pd.DataFrame([{
                'path_label': tracking_dictionary[vehicle_id]['label'],
                'outcome': tracking_dictionary[vehicle_id]['is_crash?'],
            }],
                columns=['path_label', 'outcome'])

            concated_df = pd.concat([transformed_df, df_EE, df_label], axis=1)

        final_df = pd.concat([final_df, concated_df], ignore_index=True)

    return final_df

def path_batch():
    return None