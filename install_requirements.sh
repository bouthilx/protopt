#!/usr/bin/env bash

PROJECT=$1
PROJECT_PATH=$2

WHEEL_DIR=$HOME/wheels/$PROJECT

echo "debug: install from wheels"
pip install --use-wheel --no-index --find-links=$WHEEL_DIR subprocess32
pip install --use-wheel --no-index --find-links=$WHEEL_DIR smart_dispatch
pip install --use-wheel --no-index --find-links=$WHEEL_DIR pymongo gitpython
pip install --use-wheel --no-index --find-links=$WHEEL_DIR sacred 
pip install --use-wheel --no-index --find-links=$WHEEL_DIR numpy scipy scikit-optimize

if [ ${PROJECT_PATH} -z ]
then
    bash ${PROJECT_PATH}/install_requirements.sh
fi
