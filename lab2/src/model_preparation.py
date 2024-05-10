from joblib import dump, load
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from pathlib import Path

path = Path('data')
df_train_transformed = load(path / 'preprocessing/train/data.joblib.gz')

X_train, y_train = df_train_transformed.drop(columns=['price']), df_train_transformed['price']

model = SGDRegressor(random_state=42)
model.fit(X_train, y_train)

dump(model, path / 'model.joblib.gz')

train_score = r2_score(y_train, model.predict(X_train))
print('Trained model. R2 for train data', train_score)
