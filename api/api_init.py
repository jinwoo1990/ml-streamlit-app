import os
import helper


DB_DATABASE_NAME = os.environ.get('MONGO_INITDB_DATABASE', '')
DB_MODEL_COLLECTION_NAME = os.environ.get('MONGO_MODEL_COLLECTION_NAME', '')


def init():
    """
    streamlit application 기본 작동을 위한 initial 모델 로드 함수

    :return: None
    """
    helper.load_and_save_base_model_from_pickle(DB_DATABASE_NAME, DB_MODEL_COLLECTION_NAME)


if __name__ == '__main__':
    init()
