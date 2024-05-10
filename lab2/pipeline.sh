#!/bin/bash

python3 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
python src/data_creation.py && \
python src/model_preprocessing.py && \
python src/model_preparation.py && \
python src/model_testing.py && \
echo "Succesfully completed!" || \
(echo "Failed!"; exit 1;)
