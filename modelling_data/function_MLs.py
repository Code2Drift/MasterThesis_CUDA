import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.dummy import DummyRegressor


################################################

#####              FUNCTION               ######

################################################

def load_masterfile(path:str)->pd.DataFrame:
  df = pd.read_csv(path)
  df.drop(columns=['Unnamed: 0'], inplace=True)
  print("Original dataframe shape:", df.shape)

  return df

  # df_nodupes = df.drop_duplicates(keep='last')
  # print("\nDeleted duplicate:", df.duplicated().sum())
  # print(f"Shape after with no duplicate: {df_nodupes.shape}")
  #
  # return df_nodupes


def split_train_test(input: pd.DataFrame,
                     output: pd.DataFrame,
                     test_size: float,
                     seed: int) -> pd.DataFrame:
  """
  :param input: input features
  :param output: output features
  :param float test_size: distribution of test batch from 0 - 1
  """
  from sklearn.model_selection import train_test_split

  # first batch Train and temp
  X_train, X_temp, y_train, y_temp = train_test_split(input, output,
                                                              test_size=test_size,
                                                              random_state=seed)
  return X_train, X_temp, y_train, y_temp

def train_model(estimator:BaseEstimator, input, output):
  estimator.fit(input, output)

def evaluate_model(estimator:BaseEstimator, in_train, out_train, in_val, out_val):
  # training
  y_train_predict = estimator.predict(in_train)
  rmse_train = mean_squared_error(out_train, y_train_predict)**0.5

  # validation
  y_valid_predict = estimator.predict(in_val)
  rmse_valid = mean_squared_error(out_val, y_valid_predict)**0.5

  return rmse_train, rmse_valid

def predict_data(X_train, y_train, algo):
    algo.fit(X_train, y_train)
    prediction = algo.predict(X_train)

    print(classification_report(y_train, prediction))
    return prediction


def to_numBoolean(dataframe, column_list):
  for column in column_list:
    dataframe[column] = dataframe[column].astype(int)

  return dataframe


def split_train_test(input: pd.DataFrame,
                     output: pd.DataFrame,
                     test_size: float,
                     seed: int) -> pd.DataFrame:
  """
  :param input: input features
  :param output: output features
  :param float test_size: distribution of test batch from 0 - 1
  """
  from sklearn.model_selection import train_test_split

  # first batch Train and temp
  X_train, X_temp, y_train, y_temp = train_test_split(input, output,
                                                      test_size=test_size,
                                                      random_state=seed)
  return X_train, X_temp, y_train, y_temp
