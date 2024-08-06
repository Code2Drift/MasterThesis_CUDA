"""
Iterate target folder and extract the result
"""
import os
import pandas as pd
from pre_process import data_process
from pathlib import Path
import yaml
import time

'''
Main file iteration in a folder:

1. create absolute path
2. process file using YoFlow
3. preprocess YoFlow signal and convert into readable dataframe
4. iterate over files in a folder 
5. export main dataframe into a csv file
'''


"""
Load necessary object
"""
main_path = Path(__file__).parent.absolute()
config_path = os.path.join(main_path, 'config.yaml')


### Path object
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# main_path = config['main_path']['Work']
main_path = config['main_path']['Home']

"""
Define source and target path
"""

## 1. Define Target Batch
target_batch = "06BATCH_SC5_Wagon"

full_target_batch = target_batch + "/"
# batch_path = os.path.join(config['yoflow_source_home']['scenario2'], full_target_batch)
# batch_path = os.path.join(config['yoflow_source_ssd']['scenario4'], full_target_batch)
batch_path = os.path.join(config['additional_data']['sc5'], full_target_batch)


"""
Initatior and reference objects
"""
main_df = pd.DataFrame()
start_time = time.time()
iteration = 1

for target_name in os.listdir(batch_path):
    temp_path1 = os.path.join(batch_path, target_name)

    main_df = pd.DataFrame()

    for file_name in os.listdir(temp_path1):

        if file_name.endswith('.mp4'):

            temp_path2 = os.path.join(temp_path1, file_name)
            temp_path2 = temp_path2.replace('\\', '/')

            target = f" iter: {iteration}"

            YOFLOW_dictionary = data_process.YOFLOW_main(video_path=temp_path2, target_name=temp_path2)

            result_df = data_process.process_defaultdict(YOFLOW_dictionary)

            main_df = pd.concat([main_df, result_df], ignore_index=True)

            iteration += 1
            temp_path2 = temp_path1

    csv_name = temp_path1.split('/')[-1]  + ".csv"
    result_path = os.path.join(main_path, config['additional_data']['target'], csv_name)

    main_df.to_csv(result_path, index=False)



end_time = time.time()


print(" ")
print(f"Folder iteration needed: {round((end_time - start_time)/60, 2)} mins")
print(f"saved in: {result_path}")