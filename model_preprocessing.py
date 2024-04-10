import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from pathlib import Path
import os

path = Path(__file__).resolve().parent / 'data/raw'
df_train = pd.read_csv(path / 'train/data.csv')
df_test = pd.read_csv(path / 'test/data.csv')

# Удаляем строки с пустыми значениями
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# Удаляем строки с выбросами
query = "10 < area < 200 & 1 <= floor <= 20 & 1900 < year < 2025" + \
    " & 0 < distance < 10 & 500 < price < 50000"
df_train.query(query, inplace=True)
df_test.query(query, inplace=True)

preprocessors = ColumnTransformer(transformers=[
    ('area', Pipeline([
        ('scaler', MinMaxScaler())
    ]), ['area']),
    ('floor', Pipeline([
        ('scaler', MinMaxScaler())
    ]), ['floor']),
    ('year', Pipeline([
        ('scaler', MinMaxScaler())
    ]), ['year']),
    ('distance', Pipeline([
        ('scaler', MinMaxScaler())
    ]), ['distance']),
    ('price', 'passthrough', ['price'])
], verbose_feature_names_out=False
)
preprocessors.set_output(transform='pandas')

df_train_transformed = preprocessors.fit_transform(df_train)
df_test_transformed = preprocessors.transform(df_test)

train_path = Path(__file__).resolve().parent / 'data/preprocessing/train'
test_path = Path(__file__).resolve().parent / 'data/preprocessing/test'
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

dump(preprocessors, train_path.parent / 'preprocessors.joblib.gz')
dump(df_train_transformed, train_path / 'data.joblib.gz')
dump(df_test_transformed, test_path / 'data.joblib.gz')

print('Preprocessed', df_train.shape[0], 'train rows and',
      df_test.shape[0], 'test rows')
