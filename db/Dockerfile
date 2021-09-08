FROM mongo:4-xenial
WORKDIR /data/raw/
ENV TZ=Asia/Seoul
COPY ./train.csv /data/raw/train.csv
COPY ./sample_model.json /data/raw/sample_model.json
COPY ./db_init.sh /docker-entrypoint-initdb.d/