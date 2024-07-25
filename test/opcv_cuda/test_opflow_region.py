import cv2 as cv
import numpy as np

from src import utils
from pathlib import Path
import yaml
import os

"""
Path Configuration
"""

main_path = Path(__file__).parent.parent.parent.absolute()
config_path = os.path.join(main_path, 'config.yaml')

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

## main path
main_path = config['main_path']['Home']
# dump_path = config['test_data']['dump_target']

## assign video path
video_path = config['test_data']['normal_SC2']
video_path = os.path.join(main_path, video_path)
cap = cv.VideoCapture(video_path)

frame_1, _ = utils.show_frames(180, cap)
frame_1 = utils.resize_frame(frame_1, 480)

x1, y1 = 0, 0
x2, y2 = 700, 80
frame_1[y1:y2, x1:x2] = 0

x11, y11 = 0, 80
x21, y21 = 200, 180
frame_1[y11:y21, x11:x21] = 0

x13, y13 = 700, 0
x23, y23 = 854, 40
frame_1[y13:y23, x13:x23] = 0

polygon_points1 = np.array([[854, 120], [854, 250], [740, 200]])
polygon_points2 = np.array([[360, 80], [500, 135], [700, 80]])

# Fill the polygon with black
cv.fillPoly(frame_1, [polygon_points1], (0, 0, 0))
cv.fillPoly(frame_1, [polygon_points2], (0, 0, 0))


cv.imshow("frame - 1", frame_1)
# cv.imshow("frame - 2", frame_2)
cv.waitKey()
cv.destroyAllWindows()