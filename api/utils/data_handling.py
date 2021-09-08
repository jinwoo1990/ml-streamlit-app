import os
import datetime
import pymongo
import numpy as np
import pandas as pd
import boto3
from io import BytesIO
import joblib
from loggers import logger


DB_USERNAME = os.environ.get('MONGO_USERNAME', 'root')
DB_PASSWORD = os.environ.get('MONGO_PASSWORD', 'root')
DB_HOST = os.environ.get('MONGO_HOST', 'localhost')
DB_PORT = os.environ.get('MONGO_PORT', '27017')
CONN_STR = "mongodb://%s:%s@%s:%s" % (DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT)
DB_DATABASE_NAME = os.environ.get('MONGO_INITDB_DATABASE', 'ml')
DB_MODEL_COLLECTION_NAME = os.environ.get('MONGO_MODEL_COLLECTION_NAME', 'model')
DB_DATA_COLLECTION_NAME = os.environ.get('MONGO_DATA_COLLECTION_NAME', 'data')

os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-2'
os.environ['AWS_PROFILE'] = "ml-streamlit"


def write_to_s3(file, path):
    if path[:5] == 's3://':
        s3_bucket, s3_key = path.split('/')[2], path.split('/')[3:]
        s3_key = '/'.join(s3_key)
        with BytesIO() as f:
            joblib.dump(file, f)
            f.seek(0)
            boto3.client("s3").upload_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)


def read_from_s3(path):
    if path[:5] == 's3://':
        s3_bucket, s3_key = path.split('/')[2], path.split('/')[3:]
        s3_key = '/'.join(s3_key)
        with BytesIO() as f:
            boto3.client("s3").download_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)
            f.seek(0)
            file = joblib.load(f)

    return file


def load_data(source_type='db', input_path=''):
    """
    데이터 로드를 위한 함수

    :param input_path: input 경로
    :param source_type: source 데이터 유형
    :return: input 경로에서 읽어온 raw 데이터
    """
    logger.info('## Load data')
    if source_type == 'db':
        client = pymongo.MongoClient(CONN_STR)
        database = client[DB_DATABASE_NAME]
        collection = database[DB_DATA_COLLECTION_NAME]
        cursor = collection.find()
        result = list(cursor)
        raw_data = pd.DataFrame(result)
        del raw_data['_id']
        raw_data = raw_data.replace('', np.nan, regex=True)
    elif source_type == 'csv':
        raw_data = pd.read_csv(input_path, sep=',')
    else:
        logger.error("source_type not recognized: should be 'db' or 'csv'")
        raise ValueError

    logger.info('raw_data: \n %s' % raw_data.head())

    return raw_data


def save_model_to_db(model_objects):
    """
    모델을 db 에 저장하기 위한 함수

    :param model_objects: 저장할 model objects
    :return: None
    """
    client = pymongo.MongoClient(CONN_STR)
    database = client[DB_DATABASE_NAME]
    collection = database[DB_MODEL_COLLECTION_NAME]

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
                                  'eval_results': model_objects['eval_results'],
                                  'fitted_model': model_objects['fitted_model'],
                                  'explainer': model_objects['explainer']})

    logger.info('The model %s saved successfully!' % info.inserted_id)


def load_model_from_db(model_name):
    """
    db 에서 원하는 모델을 불러오기 위한 함수

    :param model_name: 가져오고 싶은 모델 이름
    :return: db 에서 불러온 model objects
    """
    client = pymongo.MongoClient(CONN_STR)
    database = client[DB_DATABASE_NAME]
    collection = database[DB_MODEL_COLLECTION_NAME]

    cursor = collection.find({'model_name': model_name}).sort("created_timestamp", -1).limit(1)
    model_objects = cursor.next()

    return model_objects
