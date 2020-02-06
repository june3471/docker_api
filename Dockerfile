FROM tensorflow/tensorflow:nightly-py3
MAINTAINER ifu_seok@naver.com
RUN pip3 install flask pillow
EXPOSE 5000
