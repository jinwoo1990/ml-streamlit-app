FROM mongo:4-xenial
WORKDIR /data/raw/
COPY ./train.csv /data/raw/
COPY ./db_init.sh /docker-entrypoint-initdb.d/