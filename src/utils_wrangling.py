import pandas as pd
import numpy as np
import dill as pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pre_process.signal_process import  butter_LPF_Viz as signal_process

def to_dataframe_OneObject(pickle_object, int_obj):
    ## convert OF_mag to optical flow bins
    df_opflow2 = pd.DataFrame(pickle_object[int_obj]['OF_mag'],
                              columns=[
                                  'bins_0', 'bins_1', 'bins_2', 'bins_3',
                                  'bins_4', 'bins_5', 'bins_6', 'Bins_7'])
    ## create instance of Frame series
    series_frame = pd.DataFrame(pickle_object[int_obj]['Frame'], columns=['Frame'])

    ## process signal filtering
    filtered_opflow = signal_process(df_opflow2, cutoff=2, fs=60, order=1)

    ## combine bins dataframe with frame series
    filtered_opflow = pd.concat([filtered_opflow, series_frame], axis=1)

    return filtered_opflow


def visualize_opflow(dataframe):
    labels = ['0-45°', '315-360°', '270-315°', '225-270°',
              '180-225°', '135-180°', '90-135°', '45-90°']

    for i in range(dataframe.shape[1]):
        columns = dataframe.columns[i]
        if columns == 'Frame':
            break
        else:
            sns.lineplot(data=dataframe, x='Frame', y=dataframe[columns], label=labels[i])

    plt.xlabel('Frame')
    plt.ylabel('Magnitude')
    plt.title('HOOF Magnitude Over Frames')
    plt.legend(title='Angle Bins')
    plt.show()

def process_to_dataframe(load_tracking, processtarget_path, ):
    """

    :param load_tracking:
    :return:
    """
    final_df = pd.DataFrame()

    for vec_id, _ in load_tracking.items():
        ## remove detection with less than
        if len(load_tracking[vec_id]['Frame']) < 30:
            print(vec_id, 'removed')

        else:

            '''  process optical flow histogram '''
            ## create df for processing optical flow
            df_opflow = pd.DataFrame(load_tracking[vec_id]['OF_mag'],
                                     columns=[
                                         'bins_0', 'bins_1', 'bins_2', 'bins_3',
                                         'bins_4', 'bins_5', 'bins_6', 'bins_7'])

            ## filter OF signal
            filtered_opflow = signal_process(df_opflow, cutoff=2, fs=60, order=1)
            quantiles = filtered_opflow.quantile([0.5, 0.75])
            df_transform = {}

            ## Iterate over each column to collect 0.5 & 0.75 quantiles
            for column in filtered_opflow.columns:
                df_transform[f'{column}_50'] = round(quantiles.loc[0.5, column], 0)
                df_transform[f'{column}_75'] = round(quantiles.loc[0.75, column], 0)

            ## Convert the dictionary to a DataFrame
            transformed_df = pd.DataFrame([df_transform])

            '''  process exit entry assessment '''
            df_EE = pd.DataFrame([
                {
                    'Entry': load_tracking[vec_id]['Entry'],
                    'Entry_point': load_tracking[vec_id]['Entry_point'],
                    'Exit': load_tracking[vec_id]['Exit'],
                    'Exit_point': load_tracking[vec_id]['Exit_point']
                }
            ], columns=['Entry', 'Entry_point', 'Exit', 'Exit_point'])

            '''  process data label '''
            df_label = pd.DataFrame([{
                'path_label': load_tracking[vec_id]['Label'],
                'outcome': load_tracking[vec_id]['is_crash?'],
            }],
                columns=['path_label', 'outcome'])

            concated_df = pd.concat([transformed_df, df_EE, df_label], axis=1)

        final_df = pd.concat([final_df, concated_df], ignore_index=True)

    final_df.to_csv(r'E:\01_Programming\Py\Masterarbeit_BeamNG\data_extract\YoFlow_res\processed_df.csv', index=False)