$DOCKER_NAME = "mnist-tensoorflow"
$REPOSITORY_NAME = "guobowen1990/mnist-pipeline"
$TAG = "latest"
$RUN_TEST = 1

if ($RUN_TEST) {
    cd ..\test
    .\test_pipeline.ps1
    cd ..\common_docker
}

mkdir ./src
cp ../src/* ./src/
docker pull zzn2/tensorflow-with-conda
docker build -t $DOCKER_NAME .
Remove-Item -Force -Recurse ./src
docker tag $DOCKER_NAME ${REPOSITORY_NAME}:${TAG}
docker push $REPOSITORY_NAME
