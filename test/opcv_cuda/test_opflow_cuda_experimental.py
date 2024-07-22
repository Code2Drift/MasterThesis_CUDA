import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from src import utils
import dill as pickle
import pandas as pd
pd.set_option('display.max_columns', 500)

"""
Path Configuration
"""
main_path = utils.PATH_back_two_levels()

''' 
Video Configuration 
'''
#vid_path = r'E:\01_Programming\Py\MasterThesis_CUDA\test_dataset\single_file\NT-70-40.mp4'
vid_path = r"E:\01_Programming\Py\MasterThesis_CUDA\test_dataset\single_file\NT-36-56.mp4"
print(vid_path)
cap = cv.VideoCapture(vid_path)
resolution = (854, 480)


''' 
YOLO Configuration 
'''
yolov8m_path = r'E:\01_Programming\Py\MasterThesis_CUDA\yolo_models\yolov8m.pt'
print(yolov8m_path)
model = YOLO(yolov8m_path)
model.to('cuda')


'''
Entry Exit Intialization Params
'''
poly_1 = np.array([[118, 200], [148, 190], [155, 450], [67, 450]])   ## left lane
poly_2 = np.array([[250, 150], [411, 130], [425, 135], [255, 160]])  ## upper lane
poly_3 = np.array([[520, 135], [530, 130], [730, 190], [727, 200]])  ## right lane
poly_4 = np.array([[850, 280], [850, 475], [650, 475]])              ## lower lane

polys = [poly_1, poly_2, poly_3, poly_4]


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
track_hist = defaultdict(lambda: {
    'Frame': [],
    'Entry': False,
    'Entry_point': None,
    'Exit': False,
    'Exit_point': None,
    'OF_mag': [],
    'is_crash?': False,
    'label': [],
    'last_poly': None

    })



""" Initator FRAMES """
status_success, init_frame = cap.read()

if status_success is not True:
    print("fail to read frame")

else:

        ########### initialization frame


        """ OPCV-YOLO Block """
        init_frame = utils.resize_frame(init_frame, 480)

        """ CUDA BLOCK """
        CUDA_frame_prev = cv.cuda_GpuMat()
        CUDA_frame_prev.upload(init_frame)
        CUDA_frame_prev = cv.cuda.resize(CUDA_frame_prev, resolution)
        CUDA_frame_prev = cv.cuda.cvtColor(CUDA_frame_prev, cv.COLOR_BGR2GRAY)




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

            frame_count += 1

            """ Pre-Processing """
            CUDA_frame_curr = cv.cuda_GpuMat()
            CUDA_frame_curr.upload(current_frame)
            CUDA_frame_curr = cv.cuda.resize(CUDA_frame_curr, resolution)
            CUDA_frame_curr = cv.cuda.cvtColor(CUDA_frame_curr, cv.COLOR_BGR2GRAY)
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
                # YOLO_ANNOT = YOLO_RESULT[0].plot(line_width=1, labels=False, probs=False, conf=False)


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
                    # cv.circle(YOLO_ANNOT, center=center_point, radius=3, thickness=2, color=(255, 255, 255))

                    ''' Intersection Entry Exit Assessment '''

                    ### Ver.1
                    # for idx, poly in enumerate(polys):
                    #     ## check if object center point is inside polygone
                    #     if utils.check_center_location(poly, center_point):
                    #         if not track_hist[track_id]['Entry']:
                    #             ## Set the first polygon index as the entry point
                    #             track_hist[track_id]['Entry_point'] = idx + 1
                    #             track_hist[track_id]['Entry'] = True
                    #             track_hist[track_id]['last_poly'] = idx
                    #         else:
                    #             ## If a new polygon is detected and it's different from the entry polygon
                    #             if track_hist[track_id]['last_poly'] is not None and track_hist[track_id][
                    #                 'last_poly'] != idx:
                    #                 track_hist[track_id]['Exit_point'] = idx + 1
                    #                 track_hist[track_id]['Exit'] = True
                    #                 # Optionally reset last_poly if no further tracking is needed
                    #                 track_hist[track_id]['last_poly'] = None


                    ### Ver. 2 EE-Assessment
                    start_ee_ass = time.time()
                    for idx, polygon in enumerate(polys):
                        # Check if the object's center point is inside the polygon
                        if utils.check_center_location(polygon, center_point):
                            tracking_info = track_hist[track_id]

                            # Initialize entry point if not already set
                            if not tracking_info.get('Entry', False):
                                tracking_info['Entry_point'] = idx + 1
                                tracking_info['Entry'] = True
                                tracking_info['last_poly'] = idx
                            else:
                                # Update exit point if the polygon is different from the last encountered
                                last_poly = tracking_info.get('last_poly')
                                if last_poly is not None and last_poly != idx:
                                    tracking_info['Exit_point'] = idx + 1
                                    tracking_info['Exit'] = True
                                    # Reset last_poly if no further tracking is needed (optional)
                                    tracking_info['last_poly'] = None

                    # calc EE processing time
                    end_ee_ass = time.time()
                    time_EE_assessment.append(end_ee_ass - start_ee_ass)




                    """ Optical Flow """

                    start_OF_time = time.time()
                    # create optical flow instance
                    cuda_flow = cv.cuda_FarnebackOpticalFlow.create(
                        10,          # num levels, prev = 5
                        0.5,        # pyramid scale
                        True,       # Fast pyramid
                        15,         # winSize
                        10,         # numIters, prev= 10
                        5,          # polyN
                        1.1,        # PolySigma, prev =1.1
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

                    ROI_ang = utils.cut_OF(angle, pt_1, pt_2)
                    ROI_mag = utils.cut_OF(mag_flow, pt_1, pt_2)
                    end_OF_time = time.time()
                    time_opflow.append(end_OF_time - start_OF_time)

                    ### Oragnizing flow
                    #histogram_bins = utils.HOOF_sum(ROI_mag, ROI_ang)
                    start_org_flow = time.time()

                    histogram_bins = utils.HOOF_sum_experimental(ROI_mag, ROI_ang)
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
            # cv.putText(YOLO_ANNOT, f"{fps:.2f} FPS", (440, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv.imshow("Cuda Frame", YOLO_ANNOT)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

cv.destroyAllWindows()
cap.release()


"""
Data labeling 
"""
start_labeling_process = time.time()
path_label = utils.data_labeling(vid_path)

for track_id in track_hist.keys():
    if path_label.startswith('NT'):
        track_hist[track_id]['label'] = [path_label]
        track_hist[track_id]['is_crashed'] = False
    else:
        track_hist[track_id]['label'] = [path_label]
        track_hist[track_id]['is_crashed'] = True

## calc labeling processing time
end_labeling_process = time.time()
time_labeling.append(end_labeling_process - start_labeling_process)

end_runtime = time.time()
time_full_runtime.append(end_runtime - start_runtime)

"""
serialize result
"""
# with open(r'E:\01_Programming\Py\MasterThesis_CUDA\test_dataset\dump_file\cuda_tracking_home_experimental.pickle', 'wb') as file:
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


