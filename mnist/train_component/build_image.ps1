$DOCKER_NAME = "traincnn"
$REPOSITORY_NAME = "guobowen1990/mnist-train"
$TAG = "latest"

cp ../src/* ./src/
docker build -t $DOCKER_NAME .
Remove-Item -Force -Recurse ./src
docker tag $DOCKER_NAME ${REPOSITORY_NAME}:${TAG}
docker push $REPOSITORY_NAME
