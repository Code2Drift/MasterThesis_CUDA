import pandas as pd

import pandas as pd
import os
import yaml
from pathlib import Path

BASE_DIR = Path.cwd()
BASE_DIR = str(BASE_DIR).replace("\\", "/") + "/"
print(BASE_DIR)


def load_csv(root_path: Path,
             csv_name: str,
             config_path: Path,
             define_scenario: str) -> pd.DataFrame:
    """

    :param root_path:
    :param csv_name:
    :param config_path:
    :param define_scenario:
    :return:
    """
    config_path = os.path.join(root_path, config_path)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    ## define csv target
    csv_folder = os.path.join(root_path, config['yoflow_target'][define_scenario])
    collision_partner = csv_name + ".csv"
    csv_full_path = csv_folder + collision_partner
    print(f"CSV file for {csv_name}")

    dataframe = pd.read_csv(csv_full_path)

    return dataframe


def NM2_False(dataframe):

    index2change = dataframe[dataframe['path_label'].str.startswith('NM')].index

    dataframe.loc[index2change, 'outcome'] = False

    return dataframe