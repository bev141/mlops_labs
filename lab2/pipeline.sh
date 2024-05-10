#!/bin/bash

python src/data_creation.py && \
python src/model_preprocessing.py && \
python src/model_preparation.py && \
python src/model_testing.py && \
echo "Succesfully completed!" || \
(echo "Failed!"; exit 1;)
