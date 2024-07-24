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

main_path = config['main_path']['Work']

"""
Define source and target path
"""
# folder_path = os.path.join(main_path, config['test_folder']['test_sc5'])
target = "etk800-Marshall"
full_target = target + "/"

folder_path = os.path.join(config['yoflow_source_ssd']['scenario1'], full_target)
csv_name = folder_path.split('/')[-2] + ".csv"
result_path = os.path.join(main_path, config['yoflow_target']['scenario1'], csv_name)

print(csv_name)

"""
Initatior and reference objects
"""
main_df = pd.DataFrame()
start_time = time.time()
iteration = 1

for file_name in os.listdir(folder_path):

    target = target + "--- Iter: " + iteration

    if file_name.endswith('.mp4'):

        abs_file_path = os.path.join(folder_path, file_name)

        YOFLOW_dictionary = data_process.YOFLOW_main(video_path=abs_file_path, target_name=target)

        result_df = data_process.process_defaultdict(YOFLOW_dictionary)

        main_df = pd.concat([main_df, result_df], ignore_index=True)

        iteration += 1


main_df.to_csv(result_path, index=False)
end_time = time.time()
print(f"Folder iteration needed: {round((end_time - start_time)/60, 2)} mins")
print(f"saved in: {result_path}")
