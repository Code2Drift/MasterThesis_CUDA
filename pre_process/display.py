import pandas as pd
import numpy as np
import plotly.graph_objects as go
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

def generate_timestamps(df_copy, FPS_count):
    df = deepcopy(df_copy)
    if isinstance(df, pd.DataFrame):
      num_frame = df.shape[0]
      list_idx = []
      tot_length = float(num_frame / FPS_count)
      for idx, rows in df_copy.iterrows():
          time_stamp = (tot_length * (idx+1)) / num_frame
          list_idx.append(time_stamp)
      df_copy['index'] = list_idx
      df_copy.set_index('index', inplace=True)
      df_pd = df_copy

    elif isinstance(df, np.ndarray):
      df_np = pd.DataFrame(df)
      num_frame = df_np.shape[0]
      list_idx = []
      tot_length = float(num_frame / FPS_count)
      for idx, rows in df_np.iterrows():
          time_stamp = (tot_length * (idx+1)) / num_frame
          list_idx.append(time_stamp)
      df_np['index'] = list_idx
      df_np.set_index('index', inplace=True)
      df_pd = df_np

    else:
      print("Check data type, has to be either DF or Array")

    return df_pd


def generate_visualization(data, fps_count, viz_string):
    # Generate timestamps
    my_data = generate_timestamps(data, fps_count)

    # Generate a Figure
    fig = go.Figure()

    # Loop through each column (excluding 'index' column if it's present as column)
    for column in my_data.columns:
        fig.add_trace(go.Scatter(x=my_data.index, y=my_data[column], mode='lines', name=f"{column}"))

    # Update layout
    fig.update_layout(
        xaxis_title="Timestamp [s]",
        yaxis_title="Sum of Flow Magnitude [Pixel]",
        title=f"Flow Magnitude Time Progression - {viz_string}",
        hovermode="x"
    )
    fig.show()


def plot_box(df_list):
  bin_pal = {'Bin_0': 'red', 'Bin_1': 'blue', 'Bin_2': 'green', 'Bin_3': 'yellow', 'Bin_4': 'purple', 'Bin_5':'gold', 'Bin_6':'gray', 'Bin_7':'pink'}

  fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
  axes = ax.flatten()

  for i, df in enumerate(df_list):
      df_drop = df.drop(columns=["YOLO_ID", "Frame"])
      sns.boxplot(data=df_drop, ax=axes[i], palette=bin_pal)
      axes[i].set_ylabel("Sum of Flow Magnitude [Pixel]")
      axes[i].set_xlabel("Distribution of Bins")

  plt.tight_layout()
  plt.show()