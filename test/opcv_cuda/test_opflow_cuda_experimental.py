import os
from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from src import utils
import dill as pickle
import pandas as pd
pd.set_option('display.max_columns', 500)
import yaml

"""
Path Configuration
"""

main_path = Path(__file__).parent.parent.parent.absolute()
config_path = os.path.join(main_path, 'config.yaml')

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

## main path
main_path = config['main_path']['Home']
dump_path = config['test_data']['dump_target']

## assign video path
video_path = config['test_data']['near_miss']
video_path = os.path.join(main_path, video_path)

## assign yolo detector path
yolo_models = config['YOLO']['yolo8_m']
yolo_models = os.path.join(main_path, yolo_models)
print(yolo_models)

''' 
Video Configuration 
'''
cap = cv.VideoCapture(video_path)
status, frame = cap.read()
resolution = (854, 480)


''' 
YOLO Configuration 
'''
model = YOLO(yolo_models)
model.to('cuda')


'''
Entry Exit Intialization Params
'''
polys = utils.load_EE_params()


"""
Calculate Processing Time
"""
time_full_runtime = []
time_fullprocess = []
avg_fps = []
time_preprocessing = []
time_opflow = []
time_orgflow = []
time_EE_assessment = []
time_labeling = []
time_yolo_process = []

def list_average(list):
    return round(sum(list) / len(list), 5)

start_runtime = time.time()

'''
Tracking Information
'''
frame_count = 0
track_hist = utils.load_defaultdict()


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




        while True:

            ## stop process if frame did not red correctly
            start = time.time()
            start_preprocessing = time.time()
            """ Read First Frame """
            success, current_frame = cap.read()

            if not success:
                print("video ended")
                break

            """ OPCV Block for second frame """
            current_frame = utils.resize_frame(current_frame, 480)
            current_frame = utils.cut_frame(current_frame)

            frame_count += 1

            """ Pre-Processing """
            CUDA_frame_curr = utils.send2cuda(current_frame)

            ## calc time preprocess
            end_preprocessing = time.time()
            time_preprocessing.append(end_preprocessing - start_preprocessing)

            ''' Main YO_FLOW Block '''

            start_yolo = time.time()
            YOLO_RESULT = model.track(current_frame, persist=True, conf=0.3)

            YOLO_ANNOT = current_frame.copy()

            ## calc time yolo processing
            end_yolo = time.time()
            time_yolo_process.append(end_yolo - start_yolo)

            if YOLO_RESULT[0].boxes.id is not None:

                ## plot box and yolo id on annotated frame
                YOLO_bb = YOLO_RESULT[0].boxes.xywh
                YOLO_trackID = YOLO_RESULT[0].boxes.id.numpy().astype(int)

                ## for visualization purpose
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

                    ## for visualization purpose
                    cv.circle(YOLO_ANNOT, center=center_point, radius=3, thickness=2, color=(255, 255, 255))

                    ''' Intersection Entry Exit Assessment '''

                    ### Ver. 2 EE-Assessment
                    start_ee_ass = time.time()

                    utils.EE_Assessment(polys, center_point, default_dict=track_hist,
                                        vehicle_id=track_id)

                    # calc EE processing time
                    end_ee_ass = time.time()
                    time_EE_assessment.append(end_ee_ass - start_ee_ass)


                    """ Optical Flow """

                    start_OF_time = time.time()

                    ROI_mag, ROI_ang = utils.cuda_opflow(
                        cuda_prev   =CUDA_frame_prev,
                        cuda_current=CUDA_frame_curr,
                        pt_1=pt_1,
                        pt_2=pt_2
                    )

                    end_OF_time = time.time()
                    time_opflow.append(end_OF_time - start_OF_time)

                    ### Oragnizing flow
                    start_org_flow = time.time()

                    histogram_bins = utils.HOOF_sum_experimental(ROI_mag, ROI_ang)
                    # histogram_bins = utils.HOOF_median(ROI_mag, ROI_ang)
                    track_hist[track_id]['OF_mag'].append(histogram_bins)

                    end_org_flow = time.time()
                    time_orgflow.append(end_org_flow - start_org_flow)



            init_frame = current_frame
            CUDA_frame_prev = CUDA_frame_curr

            end = time.time()
            fps = 1 / (end - start)

            avg_fps.append(fps)
            time_fullprocess.append(end - start)


            # visualization Frame-by-Frame plot of image
            YOLO_ANNOT = utils.draw_lane_area(YOLO_ANNOT)
            cv.putText(YOLO_ANNOT, f"{fps:.2f} FPS", (440, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow("Cuda Frame", YOLO_ANNOT)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

cv.destroyAllWindows()
cap.release()


"""
Data labeling 
"""
start_labeling_process = time.time()
path_label = utils.data_labeling(video_path)
crash_status = not path_label.startswith('NT')

for track_id in track_hist.keys():
    track_hist[track_id]['label'] = path_label
    track_hist[track_id]['is_crash?'] = crash_status



## calc labeling processing time
end_labeling_process = time.time()
time_labeling.append(end_labeling_process - start_labeling_process)

end_runtime = time.time()
time_full_runtime.append(end_runtime - start_runtime)

"""
serialize result
"""
# utils.serialize_data(track_hist, 'test_target.pickle', test_path=dump_path)
# with open(r'E:\01_Programming\Py\MasterThesis_CUDA\test_dataset\dump_file\cuda_tracking_home_crash.pickle', 'wb') as file:
#     pickle.dump(track_hist, file)

"""
Check processing time 
"""
dataframe_processing = pd.DataFrame({
    'full runtime': time_full_runtime,
    'full process': list_average(time_fullprocess),
    'avg fps': list_average(avg_fps),
    'preprocessing': list_average(time_preprocessing),
    'opflow': list_average(time_opflow),
    'org_opflow': list_average(time_orgflow),
    'EE_assessment': list_average(time_EE_assessment),
    'yolo processing': list_average(time_yolo_process),
    'labeling': time_labeling
})

print(dataframe_processing)

for vec_id in track_hist.keys():
    print(f"TRACK ID: {vec_id}, length: {len(track_hist[vec_id]['Frame'])} frames")
    print(f"Entry        : {track_hist[vec_id]['Entry']}")
    print(f"Entry point  : {track_hist[vec_id]['Entry_point']}")

    print(f"\nExit       : {track_hist[vec_id]['Exit']}")
    print(f"Exit point   : {track_hist[vec_id]['Exit_point']}")

    print(f"\nLabel      : {track_hist[vec_id]['label']}")
    print(f"crash        : {track_hist[vec_id]['is_crash?']}")
    print(" ")


"""
Check labeling result 
"""
# print(f"Average fps is: {sum(time_fullprocess) / len(time_fullprocess)}")
# for track_id in track_hist.keys():
#     print(f"{track_id} information: ")
#
#     print("\nEntry status")
#     print(f"Track {track_id} entry       : {track_hist[track_id]['Entry_point']}")
#     print(f"Track {track_id} entry status: {track_hist[track_id]['Entry']}")
#
#     print(f"\nExit status")
#     print(f"Track {track_id} exit       : {track_hist[track_id]['Exit_point']}")
#     print(f"Track {track_id} exit status: {track_hist[track_id]['Exit']}")
#
#     print(f"\nLabeling status")
#     print(f"Track {track_id} label       : {track_hist[track_id]['label']}")
#     print(f"Track {track_id} label status: {track_hist[track_id]['is_crashed']}")
#
#     print(" ")
#     print(" ")


