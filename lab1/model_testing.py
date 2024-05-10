from joblib import load
from sklearn.metrics import r2_score
import sys
from pathlib import Path

path = Path(__file__).resolve().parent / 'data'

df_test_transformed = load(path / 'preprocessing/test/data.joblib.gz')
model = load(path / 'model.joblib.gz')

test_score = r2_score(df_test_transformed['price'], model.predict(df_test_transformed.drop(columns='price')))

if test_score < 0.87:
    print('Test failed! R2 for test data:', test_score)
    sys.exit(1)

print('Test passed. R2 for test data:', test_score)
