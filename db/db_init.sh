#!/usr/bin/env bash

# mongo 사용자 추가 및 샘플 데이터 입력
mongo <<EOF
db = db.getSiblingDB("admin");
db.auth("${MONGO_INITDB_ROOT_USERNAME}", "${MONGO_INITDB_ROOT_PASSWORD}");

db.createUser({
    'user': "${MONGO_USERNAME}",
    'pwd': "${MONGO_PASSWORD}",
    'roles': [{
        'role': "dbOwner",
        'db': "${MONGO_INITDB_DATABASE}"}]
});

db = db.getSiblingDB("${MONGO_INITDB_DATABASE}");
db.test.drop();
db.test.insertMany(
    [
        {"name": "Titanic", "type": "binary classification"},
        {"name": "Porto Seguro's Safe Driver Prediction", "type": "binary classification"},
        {"name": "Costa Rican Household Poverty Level Prediction", "type": "multi-class classification"},
        {"name": "New York City Taxi Trip Duration", "type": "regression"},
        {"name": "Zillow's Home Value Prediction", "type": "regression"}
    ]
);
EOF

# csv import
# container 생성 시 .env 에서 만들어진 변수 참조
# mongoimport 로 csv import 시 null 값이 string ''로 입력됨 (처리 필요)
mongoimport --host=127.0.0.1 -u "${MONGO_USERNAME}" -p "${MONGO_PASSWORD}" \
--authenticationDatabase admin \
--db "${MONGO_INITDB_DATABASE}" \
--collection "${MONGO_DATA_COLLECTION_NAME}" \
--type csv --headerline --file "${SAMPLE_FILE_PATH}"

# 기본 모델 json import (여기 고쳐야될 것 같은데... load and save model from pickle 부터 고쳐야...)
# train 하고 sample_model.json 뽑는 법. 뽑고 id 지워줘야 함
# mongoexport --host=127.0.0.1 -u "${MONGO_USERNAME}" -p "${MONGO_PASSWORD}" --authenticationDatabase admin --collection=model --db=ml --out=sample.json
mongoimport --host=127.0.0.1 -u "${MONGO_USERNAME}" -p "${MONGO_PASSWORD}" --authenticationDatabase admin --db=ml --collection=model sample_model.json
