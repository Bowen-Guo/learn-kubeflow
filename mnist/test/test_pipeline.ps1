#!/bin/bash

$TEST_DIR="./temp_folder"

mkdir $TEST_DIR

# Load data
python ../src/load_data.py --output-data-folder-path=$TEST_DIR

# Train model
python ../src/train.py --input-data-folder-path=$TEST_DIR --number-of-steps=10 --output-model-folder-path=$TEST_DIR

# Score model
python ../src/score.py --input-learner-folder-path=$TEST_DIR --input-data-folder-path=$TEST_DIR --output-data-folder-path=$TEST_DIR

# Evaluate model
python ../src/evaluate.py --input-prediction-folder-path=$TEST_DIR --input-true-folder-path=$TEST_DIR --output-data-folder-path=$TEST_DIR

Remove-Item -Force -Recurse $TEST_DIR