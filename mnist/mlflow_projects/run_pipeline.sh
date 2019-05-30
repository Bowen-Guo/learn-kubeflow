FOLDER=./gen
mkdir $FOLDER

# Run load data
mlflow run . -e load_data -P output_data_folder_path=$FOLDER

# Run train model
mlflow run . -e train -P input_data_folder_path=$FOLDER -P number_of_steps=10 -P output_model_folder_path=$FOLDER

# Run score data
mlflow run . -e score -P input_learner_folder_path=$FOLDER -P input_data_folder_path=$FOLDER -P output_data_folder_path=$FOLDER

# Run evaluate model
mlflow run . -e evaluate -P input_prediction_folder_path=$FOLDER -P input_true_folder_path=$FOLDER -P output_data_folder_path=$FOLDER


