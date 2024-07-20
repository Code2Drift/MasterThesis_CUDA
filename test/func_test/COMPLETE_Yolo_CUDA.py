import torch.cuda
from ultralytics import YOLO
import cv2 as cv
import time
from scripts.src import utils
torch.cuda.set_device(0)


'''Test data'''
path_img = r"D:\yolo_models\unittest_pict.png"
path_vid = r'D:\BeamNG_dataset\Crash_8\etk800-LegranSE\NT-70-40.mp4'
yolo8m =  r'D:\yolo_models\yolov8m.pt'



"""Cuda testing"""
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)
    model = YOLO(yolo8m).to('cuda')
else:
    model = YOLO(yolo8m)

print(f"Device is {device}")




'''Test Picture'''
# start = time.time()
#
# result = model.predict(path_img, conf=0.5)
# annot_frame = result[0].plot()
#
# end = time.time()
# inf_time = end-start
#
# cv.putText(annot_frame, f"{inf_time:.2f} ms inferece time", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# cv.putText(annot_frame, f"device used: {device}", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)
#
# while True:
#
#     cv.imshow("test image", annot_frame)
#
#     if cv.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cv.destroyAllWindows()



'''Test Video'''
cap = cv.VideoCapture(path_vid)

while True:

    success, frame = cap.read()

    if not success:
        print("Fails to read in Frame")

    start = time.time()

    frame = utils.resize_frame(frame, 480)

    if not success:
        break

    result = model.track(frame, conf=0.5, persist=True)

    annot_frame = frame.copy()

    ## get all result from yolo prediction
    if result[0].boxes.id is not None:

        obj_bb = result[0].boxes.xywh
        track_id = result[0].boxes.id.numpy().astype(int)
        annot_frame = result[0].plot(line_width=1, labels=True)

    ## plot fps on annotated frame
    end = time.time()
    fps = 1 / (end - start)

    cv.putText(annot_frame, f"{fps:.2f} fps", (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.putText(annot_frame, f"device used: {device}", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)


    cv.imshow("test video", annot_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cv.destroyAllWindows()












