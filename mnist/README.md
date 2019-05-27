This is a example of Kubeflow pipeline.

* train_component:
  * build_image.ps1: powershell script to build and push Docker image to Dockerhub
  * Dockerfile: image Dockerfile, used to build image
  * train_component.yaml: component yaml file

* score_component:
  * The folder content is the same as the train component.
  
* src: source code 

* test: code to test source code

* mnist_pipeline.yaml: pipeline yaml file, which is generated from pipeline.py

* pipeline.py: code using kubeflow sdk to generate mnist_pipeline.yaml. The command line code is:
  `dsl-compile --py pipleline.py --output mnist_pipeline.yaml`



