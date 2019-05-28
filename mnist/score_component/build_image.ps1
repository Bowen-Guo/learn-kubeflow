$DOCKER_NAME = "scorecnn"
$REPOSITORY_NAME = "guobowen1990/mnist-score"
$TAG = "latest"

cp ../src/* ./src/
docker pull zzn2/tensorflow-with-conda
docker build -t $DOCKER_NAME .
Remove-Item -Force -Recurse ./src
docker tag $DOCKER_NAME ${REPOSITORY_NAME}:${TAG}
docker push $REPOSITORY_NAME
