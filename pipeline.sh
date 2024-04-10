#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

DIR="$( cd "$( dirname "$0" )" && pwd )"

python ${DIR}/data_creation.py && \
python ${DIR}/model_preprocessing.py && \
python ${DIR}/model_preparation.py && \
python ${DIR}/model_testing.py && \
echo -e "${GREEN}Succesfully completed!${NC}" || \
echo -e "${RED}Failed!${NC}"
