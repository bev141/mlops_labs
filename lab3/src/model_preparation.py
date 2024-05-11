import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
from pathlib import Path
import sys

# load dataset
iris = datasets.load_iris()

# prepare train and test data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

path = Path('data')
os.makedirs(path, exist_ok=True)

# save train and test data
np.save(path / 'train_data', X_train)
np.save(path / 'train_target', y_train)
np.save(path / 'test_data', X_test)
np.save(path / 'test_target', y_test)

# create pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', RandomForestClassifier(n_estimators=9, random_state=42)),
])

# train model
pipeline.fit(X_train, y_train)

# test model
test_score = f1_score(y_test, pipeline.predict(X_test), average='micro')
if test_score < 0.9:
    print('Test failed! f1 for test data:', test_score)
    sys.exit(1)

print('Test passed. f1 for test data:', test_score)

# save model
dump(pipeline, path / 'pipeline.joblib.gz')
