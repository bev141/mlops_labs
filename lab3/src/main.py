from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

class Input(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

path = Path('data')
pipeline = load(path / 'pipeline.joblib.gz')

app = FastAPI()

@app.post("/predict/")
async def process(input: Input):
    """Предсказывает класс ириса."""
    res = pipeline.predict([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])[0]
    classes = ['setosa', 'versicolor', 'virginica']
    return {'class': classes[res]}

@app.get("/")
async def index():
    return 'For iris classification send POST request to /predict/'
