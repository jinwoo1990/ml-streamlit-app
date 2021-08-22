import os
import datetime
import pickle
import pymongo
import numpy as np
import pandas as pd
from base_logger import logger


DB_USERNAME = os.environ.get('MONGO_USERNAME', 'root')
DB_PASSWORD = os.environ.get('MONGO_PASSWORD', 'root')
DB_HOST = os.environ.get('MONGO_HOST', '')
DB_PORT = os.environ.get('MONGO_PORT', '')
CONN_STR = "mongodb://%s:%s@%s:%s" % (DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT)


def load_data_from_db(db_name, col_name):
    """
    db 에서 data 를 가져오기 위한 함수

    :param db_name: 연결할 mongoDB db 이름
    :param col_name: 연결할 mongoDB collection 이름
    :return: mongoDB 로부터 가져온 dataframe 으로 변환된 데이터
    """
    client = pymongo.MongoClient(CONN_STR)
    database = client[db_name]
    collection = database[col_name]

    cursor = collection.find()

    result = list(cursor)

    df = pd.DataFrame(result)
    del df['_id']

    return df


def load_and_save_base_model_from_pickle(db_name, col_name):
    """
    기본 pickle 모델 파일을 db 에 업로드하기 위한 함수

    :param db_name: 연결할 mongoDB db 이름
    :param col_name: 연결할 mongoDB collection 이름
    :return: None
    """
    with open('model_objects.pkl', 'rb') as handle:
        model_objects = pickle.load(handle)

    client = pymongo.MongoClient(CONN_STR)
    database = client[db_name]
    collection = database[col_name]

    now = datetime.datetime.now()
    created_time = now.strftime('%Y-%m-%d %H:%M:%S')
    created_timestamp = now.timestamp()

    info = collection.insert_one({'model_name': model_objects['model_name'],
                                  'created_time': created_time,
                                  'created_timestamp': created_timestamp,
                                  'target_dict': model_objects['target_dict'],
                                  'null_converter': model_objects['null_converter'],
                                  'label_encoder': model_objects['label_encoder'],
                                  'features_selected': model_objects['features_selected'],
                                  'optimized_params': model_objects['optimized_params'],
                                  'fitted_model': model_objects['fitted_model'],
                                  'eval_results': model_objects['eval_results'],
                                  'explainer': model_objects['explainer']})

    logger.info('The base model %s saved successfully!' % info.inserted_id)


def save_model_to_db(db_name, col_name, model_objects):
    """
    모델을 db 에 저장하기 위한 함수

    :param db_name: 연결할 mongoDB db 이름
    :param col_name: 연결할 mongoDB collection 이름
    :param model_objects: 저장할 model objects
    :return: None
    """
    client = pymongo.MongoClient(CONN_STR)
    database = client[db_name]
    collection = database[col_name]

    now = datetime.datetime.now()
    created_time = now.strftime('%Y-%m-%d %H:%M:%S')
    created_timestamp = now.timestamp()

    info = collection.insert_one({'model_name': model_objects['model_name'],
                                  'created_time': created_time,
                                  'created_timestamp': created_timestamp,
                                  'target_dict': model_objects['target_dict'],
                                  'null_converter': model_objects['null_converter'],
                                  'label_encoder': model_objects['label_encoder'],
                                  'features_selected': model_objects['features_selected'],
                                  'optimized_params': model_objects['optimized_params'],
                                  'fitted_model': model_objects['fitted_model'],
                                  'eval_results': model_objects['eval_results'],
                                  'explainer': model_objects['explainer']})

    logger.info('The model %s saved successfully!' % info.inserted_id)


def load_model_from_db(db_name, col_name, model_name):
    """
    db 에서 원하는 모델을 불러오기 위한 함수

    :param db_name: 연결할 mongoDB db 이름
    :param col_name: 연결할 mongoDB collection 이름
    :param model_name: 가져오고 싶은 모델 이름
    :return: db 에서 불러온 model objects
    """
    client = pymongo.MongoClient(CONN_STR)
    database = client[db_name]
    collection = database[col_name]

    cursor = collection.find({'model_name': model_name}).sort("created_timestamp", -1).limit(1)
    model_objects = cursor.next()

    return model_objects


def preprocess_record(data, null_converter, label_encoder, features_selected):
    """
    categorical 컬럼의 label encoding 결과를 반환하는 함수

    :param data: 전처리되지 않은 input 데이터
    :param null_converter: null 값을 처리하기 위한 categorical column 최빈값 dictionary
    :param label_encoder: categorical 값 변환을 위한 label encoder dictionary (sklearn.preprocessing.OrdinalEncoder)
    :param features_selected: feature selection 과정에서 선택된 features list
    :return: label encoding 이 완료되고 모델링에 필요한 features 만 선택된 dataframe
    """
    for col in null_converter.keys():
        # null 값 training 데이터 최빈값으로 변환
        temp = data[col].fillna(null_converter[col])
        data[col] = temp

    for col in list(label_encoder.keys()):
        # 컬럼별로 정의된 OrdinalEncoder 를 사용해 categorical 데이터 변환 (처음 나오는 값은 -1로 변환됨)
        temp = label_encoder[col].transform(np.array(data[col]).reshape(-1, 1))
        data[col] = temp

    data["Fsize"] = data["SibSp"] + data["Parch"] + 1
    data = data.drop(["SibSp", "Parch"], axis=1)

    return data[features_selected]


def predict_record(data, clf):
    """
    예측값과 예측확률을 반환하는 함수

    :param data: transform_categorical() 을 통해 전치리가 된 input 데이터
    :param clf: 예측 모델
    :return: 모델 예측값 및 예측확률
    """
    prediction = clf.predict(data)[0]
    probability = np.max(clf.predict_proba(data))

    return prediction, probability
