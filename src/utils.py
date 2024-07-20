import cv2 as cv
import numpy as np
from numba import jit, prange
import cProfile
import pstats
import io
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

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


@jit(nopython=True, parallel=True)
def HOOF_ID(magnitude, angle):
    sum = np.zeros(9)
    for idx in prange(magnitude.shape[0]):  # for each flow map, i.e. for each image pair
        for mag, ang in zip(magnitude[idx].reshape(-1), angle[idx].reshape(-1)):
            if ang >= 360:
                ang = ang - 360  # Make sure angles are within [0, 360)
            bin_idx = int( ang // 45 )
            sum[bin_idx] += mag
    rounded_sum = np.round(sum[0:8], 1)

    return rounded_sum


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

def cut_ndarray(frame, point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2

    # Ensure coordinates are within frame bounds
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)

    # Slice the frame to get the ROI
    frame_cut = frame[y1:y2, x1:x2]
    return frame_cut


def corrected_CUT_OF(ORG_OF_frame, point_1, point_2):
    height, width = ORG_OF_frame.shape[:2]
    x1, y1 = max(0, min(width, point_1[0])), max(0, min(height, point_1[1]))
    x2, y2 = max(0, min(width, point_2[0])), max(0, min(height, point_2[1]))

    # Create a mask with zeros
    mask = np.zeros_like(ORG_OF_frame, dtype=np.uint8)
    # Set the ROI to 1
    mask[y1:y2, x1:x2] = 1

    # Apply the mask to the optical flow frame
    OF_cut = ORG_OF_frame * mask
    return OF_cut

def cut_OF_cuda(OF_frame, point_1, point_2):
    # Ensure points are integers and within frame bounds
    height, width = OF_frame.size()
    x1, y1 = max(0, min(width, point_1[0])), max(0, min(height, point_1[1]))
    x2, y2 = max(0, min(width, point_2[0])), max(0, min(height, point_2[1]))

    # Create a mask with zeros and set the ROI to 1
    mask = cv.cuda_GpuMat((height, width), cv.CV_8UC1)
    mask.setTo(0)
    roi = mask.rowRange(y1, y2).colRange(x1, x2)
    roi.setTo(1)


    # Ensure mask is of the same type as OF_frame
    mask = mask.convertTo(OF_frame.type())

    # Apply the mask to the optical flow frame
    OF_cut = cv.cuda_GpuMat(OF_frame.size(), OF_frame.type())
    cv.cuda.multiply(OF_frame, mask, OF_cut)
    return OF_cut

def check_frame(vid_path, frame_num):
    cap = cv.VideoCapture(vid_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    if not success:
        print("failed to properly read frames")
    return frame

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

    vid_label = path.split('\\')[-1].split('.')[0]

    return vid_label

"""
TODO:

1. 

"""