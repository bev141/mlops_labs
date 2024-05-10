import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

n_samples = 1000

data = {
    'area': np.round(30 + 50 * np.random.rand(n_samples), 1),
    'floor': np.random.randint(1, 10, n_samples),
    'year': np.random.randint(2000, 2024, n_samples),
    'distance': np.round(0.3 + 5 * np.random.rand(n_samples), 1)
}

# генерируем цену квартиры
data['price'] = np.round(
    (
        100 * data['area']  # цена пропорциональна площади
        + 1.5 * data['year']  # чем новее, тем дороже
        - 4 * data['distance']  # чем дальше от центра, тем дешевле
        + 0.5 * data['floor']  # чем выше этаж, тем дороже(в этом городе так:))
    ) * (1 + 0.05 * np.random.randn(n_samples)),  # добавляем случайный шум
    -1).astype(int)  # округляем до десятков и сохраняем как int

# Добавляем пустые значения
data['area'][3] = np.NaN

# Добавляем выбросы
data['price'][1] = 10

df = pd.DataFrame(data)

df_train, df_test = train_test_split(df, test_size=0.3)

train_path = Path(__file__).resolve().parent / 'data/raw/train'
test_path = Path(__file__).resolve().parent / 'data/raw/test'
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

df_train.to_csv(train_path / 'data.csv')
df_test.to_csv(test_path / 'data.csv')

print('Generated', df_train.shape[0], 'train rows and',
      df_test.shape[0], 'test rows')
