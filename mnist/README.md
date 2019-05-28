This is an example of Kubeflow pipeline (load data -> train -> score -> evaluate).

* load_data_component:
  * load_data_component.yaml: component yaml file
  
* train_component:
  * train_component.yaml: component yaml file

* score_component:
  * score_component.yaml: component yaml file
  
* evaluate_component:
  * evaluate_component.yaml: component yaml file

* common_docker:
  * build_image.ps1: powershell script to build Docker image and push to Dockerhub
  * Dockerfile: Dockerfile of the image
  
* src: source code 

* test: code to test source code
  * test_pipeline.py: python script to test the full pipeline
  * test_pipeline.ps1: powershell script ot test the full pipeline

* mnist_pipeline.yaml: pipeline yaml file, which is generated from pipeline.py

* pipeline.py: code using kubeflow sdk to generate mnist_pipeline.yaml. The command line code is:

  `dsl-compile --py pipleline.py --output mnist_pipeline.yaml`



