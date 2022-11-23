FROM tensorflow/tensorflow:1.4.0-gpu-py3

WORKDIR /rgcn

COPY . .

ENV PYTHONPATH=/rgcn/code/optimization

RUN pip install theano
