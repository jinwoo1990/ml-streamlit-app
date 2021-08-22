import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import helper
import pipeline
from base_logger import logger
import traceback


app = Flask(__name__)

DB_DATABASE_NAME = os.environ.get('MONGO_INITDB_DATABASE', '')
DB_MODEL_COLLECTION_NAME = os.environ.get('MONGO_MODEL_COLLECTION_NAME', '')


@app.route('/model/predict/', methods=['POST'])
def predict():
    """
    예측 결과를 반환하는 함수

    :return: 전처리된 input 데이터, 예측 결과 및 확률을 포함한 dataframe
    """
    try:
        json_data = request.get_json()
        model_name = json_data[0]['model_name']

        model_objects = helper.load_model_from_db(DB_DATABASE_NAME, DB_MODEL_COLLECTION_NAME, model_name)

        null_converter = pickle.loads(model_objects['null_converter'])
        label_encoder = pickle.loads(model_objects['label_encoder'])
        features_selected = pickle.loads(model_objects['features_selected'])
        fitted_model = pickle.loads(model_objects['fitted_model'])

        data = pd.DataFrame(json_data)
        del data['model_name']
        data = helper.preprocess_record(data, null_converter, label_encoder, features_selected)

        prediction, probability = helper.predict_record(data, fitted_model)

        data['prediction'] = prediction
        data['probability'] = probability

        output = data.to_dict(orient='rows')[0]
        output = jsonify(output)

    except Exception:
        logger.error(traceback.format_exc())
        raise

    return output


@app.route('/model/train/', methods=['POST'])
def train():
    """
    모델을 train 시키기 위한 함수

    :return: 모델 학습 완료 확인용 success 메세지
    """
    try:
        json_data = request.get_json()
        target_col = json_data['target_col']
        model_name = json_data['model_name']
        n_folds = json_data['n_folds']
        iteration_num = json_data['iteration_num']
        eval_metric = json_data['eval_metric']
        pipeline.create_model_objects('db', target_col, model_name, n_folds, iteration_num, eval_metric)

    except Exception:
        logger.error(traceback.format_exc())
        raise

    return 'success'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
