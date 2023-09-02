FROM python:3.9

ENV BUILDVERSION=2
# RUN apt-get update
RUN apt-get update && apt-get install libgl1  -y
#install requirements
COPY ./requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip3 install -r requirements.txt

#copy app
COPY . /api
WORKDIR /api

CMD ["gunicorn", "-w", "3", "-b", ":5000", "-t", "0", "--reload", "wsgi:app"]