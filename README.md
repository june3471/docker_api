# docker_api

## Using Tensorflow Serving with Flask test
여러 개의 도커 컨테이너를 활용하여 client 입장에서 tensorflow model predict 결과를 배포하는 실습

### Dockerfile build
Dockerfile build
- docker build -t "tag name" .

You should change docker-compose.yml same to your tag name

### when using inception rest api
curl -F "file=@dog1-4.jpg" http://localhost:5000/inception/predict/

### when using yolov3 rest api
curl -F "instances=@dog1-4.jpg" -F "class=@coco.names" -H "size:416" -H "thresh:0.5" http://localhost:5000/yolov3/predict/
