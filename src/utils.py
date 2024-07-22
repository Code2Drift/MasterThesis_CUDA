import cv2 as cv
import numpy as np
from numba import jit, prange
import cProfile
import pstats
import io
from pathlib import Path
import os
import dill as pickle
from collections import defaultdict



"""
Path organizing 
"""
def get_project_root() -> Path:
    return Path(__file__).parent.parent

def PATH_back_two_levels():
    current_path = os.getcwd()
    two_levels_up = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
    return two_levels_up


"""
Code profiling 
"""
def profile(func):
    """A decorator that uses cProfile to profile a function"""
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result
    return wrapper


"""
Image / Frame Processing 
"""
#@profile
def show_frames(frame_num, cap, resolution):
    if not cap.isOpened():
        print("Error in opening video file")

    next_frame = frame_num + 1
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    status, frame_num = cap.read()
    _, next_frame = cap.read()

    return frame_num, next_frame

#@profile
def resize_frame(frame, resolution):
    select_res = {
        360 : (640, 360),
        480 : (854, 480),
        720 : (1280, 720)
    }
    tuple_resolution = select_res.get(resolution)
    frame_width, frame_height = tuple_resolution
    frame_resized = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)
    return frame_resized

def check_frame(vid_path, frame_num):
    cap = cv.VideoCapture(vid_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    if not success:
        print("failed to properly read frames")
    return frame



"""
Optical Flow Block Histogramm
"""
@jit(nopython=True, parallel=True)
def HOOF_sum(magnitude, angle):
    sum = np.zeros(9)
    for idx in prange(magnitude.shape[0]):  # for each flow map, i.e. for each image pair
        for mag, ang in zip(magnitude[idx].reshape(-1), angle[idx].reshape(-1)):
            if ang >= 360:
                ang = ang - 360  # Make sure angles are within [0, 360)
            bin_idx = int( ang // 45 )
            sum[bin_idx] += mag
    rounded_sum = np.round(sum[0:8], 1)

    return rounded_sum

@jit(nopython=True, parallel=True)
def HOOF_avg(magnitude, angle):
    sum = np.zeros(9)  # Array to store the sum of magnitudes for each bin
    count = np.zeros(9)  # Array to store the count of elements in each bin

    for idx in prange(magnitude.shape[0]):  # for each flow map, i.e., for each image pair
        for mag, ang in zip(magnitude[idx].reshape(-1), angle[idx].reshape(-1)):
            if ang >= 360:
                ang = ang - 360  # Make sure angles are within [0, 360)
            bin_idx = int(ang // 45)
            sum[bin_idx] += mag
            count[bin_idx] += 1

    # Calculate the average for each bin, handle division by zero
    average = np.zeros(9)
    for i in range(9):
        if count[i] > 0:
            average[i] = sum[i] / count[i]

    rounded_average = np.round(average[0:8], 1)

    return rounded_average


@jit(nopython=True, parallel=True, fastmath=True)
def HOOF_sum_experimental(magnitude, angle):
    bin_ranges = np.array([45, 90, 135, 180, 225, 270, 315, 360])

    sum_bins = np.zeros(9)

    for idx in prange(magnitude.shape[0]):  # for each flow map, i.e. for each image pair
        mags = magnitude[idx].reshape(-1)
        angs = angle[idx].reshape(-1)

        for i in range(mags.size):
            mag = mags[i]
            ang = angs[i]

            if ang >= 360:
                ang -= 360  # Ensure angles are within [0, 360)

            # Find the appropriate bin
            bin_idx = np.searchsorted(bin_ranges, ang)
            sum_bins[bin_idx] += mag

    rounded_sum = np.round(sum_bins[:8], 1)

    return rounded_sum


@jit(nopython=True, parallel=True)
def HOOF_median(magnitude, angle):
    num_bins = 8
    bins = np.zeros((num_bins, magnitude.shape[0] * magnitude.shape[1]), dtype=np.float32)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for idx in prange(magnitude.shape[0]):  # for each flow map, i.e. for each image pair
        mags = magnitude[idx].reshape(-1)
        angs = angle[idx].reshape(-1)

        for mag, ang in zip(mags, angs):
            if ang >= 360:
                ang = ang - 360  # Make sure angles are within [0, 360)
            bin_idx = int(ang // 45)
            bins[bin_idx, bin_counts[bin_idx]] = mag
            bin_counts[bin_idx] += 1

    # Compute median for each bin
    medians = np.zeros(num_bins)
    for i in range(num_bins):
        if bin_counts[i] > 0:
            medians[i] = np.median(bins[i, :bin_counts[i]])
        else:
            medians[i] = 0.0

    rounded_medians = np.round(medians, 1)

    return rounded_medians

"""
Optical Flow Calculation
"""
def cuda_opflow(cuda_prev, cuda_current, pt_1, pt_2):

    cuda_flow = cv.cuda_FarnebackOpticalFlow.create(
        10,  # num levels, prev = 5
        0.5,  # pyramid scale
        True,  # Fast pyramid
        15,  # winSize
        10,  # numIters, prev= 10
        5,  # polyN
        1.1,  # PolySigma, prev =1.1
        0,  # flags
    )

    cuda_flow = cv.cuda_FarnebackOpticalFlow.calc(
        cuda_flow, cuda_prev, cuda_current, None,
    )

    cuda_flow_x = cv.cuda_GpuMat(cuda_flow.size(), cv.CV_32FC1)
    cuda_flow_y = cv.cuda_GpuMat(cuda_flow.size(), cv.CV_32FC1)
    cv.cuda.split(cuda_flow, [cuda_flow_x, cuda_flow_y])

    cuda_mag, cuda_ang = cv.cuda.cartToPolar(
        cuda_flow_x, cuda_flow_y, angleInDegrees=True
    )

    angle = cuda_ang.download()
    mag_flow = cuda_mag.download()

    ROI_ang = cut_OF(angle, pt_1, pt_2)
    ROI_mag = cut_OF(mag_flow, pt_1, pt_2)

    return ROI_mag, ROI_ang


def send2cuda(frame):
    cuda_frame = cv.cuda_GpuMat()
    cuda_frame.upload(frame)
    cuda_frame = cv.cuda.resize(cuda_frame, (854, 480))
    cuda_frame = cv.cuda.cvtColor(cuda_frame, cv.COLOR_BGR2GRAY)

    return cuda_frame

#@profile
def INIT_DenseOF(firstFrame, secondFrame):
    GR_firstFrame = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)
    GR_secondframe = cv.cvtColor(secondFrame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(GR_firstFrame, GR_secondframe, None,
                                       0.5, 5, 15, 10, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
    return GR_firstFrame, GR_secondframe, flow

#@profile
def AFTER_DenseOF(firstFrame, secondFrame, opt_flow):
    GR_firstFrame = firstFrame
    GR_secondframe = cv.cvtColor(secondFrame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(GR_firstFrame, GR_secondframe, opt_flow,
                                       0.5, 5, 15, 10, 5, 1.1, 0)
    return GR_firstFrame, GR_secondframe, flow

def Oneline_OF(first_frame, second_frame):
    GR_firstFrame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    GR_secondFrame = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(GR_firstFrame, GR_secondFrame, None,
                                       0.5, 5, 15, 10, 5, 1.1, 0)
    flow = np.round(flow, 2)

    return GR_firstFrame, GR_secondFrame, flow

#@profile
def VIZ_OF_DENSE(flow, frame_2):
    mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    ang = (360 - ang) % 360
    mag = np.round(mag, 2)
    ang = ang.astype(int)
    hsv_mask = np.zeros_like(frame_2)
    hsv_mask[:, :, 1] = 255
    hsv_mask[:, :, 0] = ang / 2
    hsv_mask[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr_img = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)
    return bgr_img, mag, ang

#@profile
def cut_OF(ORG_OF_frame, point_1, point_2):
    mask = np.zeros_like(ORG_OF_frame)
    mask[point_1[1]:point_2[1], point_1[0]:point_2[0]] = 1
    OF_cut = ORG_OF_frame * mask
    return OF_cut



"""
Data labeling: EE-Assessment and data labels 
"""
def draw_lane_area(annotated_frame):
    poly_1 = np.array([[118, 200], [148, 190], [155, 450], [67, 450]])
    poly_2 = np.array([[250, 150], [411, 130], [425, 135], [255, 160]])
    poly_3 = np.array([[520, 135], [530, 130], [730, 190], [727, 200]])
    poly_4 = np.array([[850, 280], [850, 475], [650, 475]])

    cv.fillPoly(annotated_frame, pts=[poly_1], color=(255, 0, 0))
    cv.fillPoly(annotated_frame, pts=[poly_2], color=(0, 255, 0))
    cv.fillPoly(annotated_frame, pts=[poly_3], color=(0, 0, 255))
    cv.fillPoly(annotated_frame, pts=[poly_4], color=(0, 255, 255))

    return annotated_frame


def check_center_location(polygon, point):
    n = len(polygon)
    inside = False
    x, y = point

    # Loop through each edge of the polygon
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]


        if (y1 > y) != (y2 > y):
            xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x < xinters:
                inside = not inside

    return inside

def data_labeling(path):

    vid_label = path.split('/')[-1].split('.')[0]

    return str(vid_label)

def serialize_data(object, pickle_file_name, test_path):
    with open(os.path.join(test_path, pickle_file_name), 'wb') as file:
        pickle.dump(object, file)

def load_defaultdict():
    default_dictionary = defaultdict(lambda: {
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

    return default_dictionary

def load_EE_params():
    poly_1 = np.array([[118, 200], [148, 190], [155, 450], [67, 450]])  ## left lane
    poly_2 = np.array([[250, 150], [411, 130], [425, 135], [255, 160]])  ## upper lane
    poly_3 = np.array([[520, 135], [530, 130], [730, 190], [727, 200]])  ## right lane
    poly_4 = np.array([[850, 280], [850, 475], [650, 475]])  ## lower lane

    polys = [poly_1, poly_2, poly_3, poly_4]

    return polys

def EE_Assessment(polygones, center_point, default_dict, vehicle_id):
    for idx, polygon in enumerate(polygones):
        # Check if the object's center point is inside the polygon
        if check_center_location(polygon, center_point):
            tracking_info = default_dict[vehicle_id]

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