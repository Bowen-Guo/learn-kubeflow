# Run pipeline in docker
FOLDER=./temp_folder
mkdir $FOLDER

# Run load data
python /work_dir/src/load_data.py --output-data-folder-path=$FOLDER

# Run train model
python /work_dir/src/train.py --input-data-folder-path=$FOLDER --number-of-steps=10 --output-model-folder-path=$FOLDER

# Run score data
python /work_dir/src/score.py --input-learner-folder-path=$FOLDER --input-data-folder-path=$FOLDER --output-data-folder-path=$FOLDER

# Run evaluate model
python /work_dir/src/evaluate.py --input-prediction-folder-path=$FOLDER --input-true-folder-path=$FOLDER --output-data-folder-path=$FOLDER
