FROM python:3.7-slim

WORKDIR /usr/src/ispip


COPY ./requirements.txt . 
RUN pip install -r requirements.txt 

COPY . .

RUN chmod 777 /usr/src/ispip


