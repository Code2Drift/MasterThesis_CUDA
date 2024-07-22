"""
Iterate target folder and extract the result
"""
import os
import pandas as pd
from pre_process import data_process


'''
Main file iteration in a folder:

1. create absolute path
2. process file using YoFlow
3. preprocess YoFlow signal and convert into readable dataframe
4. iterate over files in a folder 
5. export main dataframe into a csv file
'''


folder_path = r"E:\Capture\BeamNG_dataset\BeamNG.drive\sample"
result_path = r"E:\01_Programming\Py\Masterarbeit_BeamNG\data_extract\YoFlow_res\FULL_RESULT_SAMPLE.csv"
yolo_8m = r"E:\01_Programming\Py\Masterarbeit_BeamNG\scripts\YOLO_mdls\yolov8m.pt"


main_df = pd.DataFrame()

for file_name in os.listdir(folder_path):

    if file_name.endswith('.mp4'):
        abs_file_path = os.path.join(folder_path, file_name)

        YoFlow_result = data_process.YoFlow_main(abs_file_path, yolo_models=yolo_8m)

        result_df = data_process.process_defaultdict(YoFlow_result)

        main_df = pd.concat([main_df, result_df], ignore_index=True)

main_df
main_df.to_csv(result_path, index=False)
