import streamlit as st
import streamlit.components.v1 as components
from collections import OrderedDict
import os
import pickle
import pymongo
import json
import datetime
import requests
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from loggers import logger


# 모델 API endpoint
url = 'http://api:5000'
endpoint = '/model/predict/'

# DB info
DB_USERNAME = os.environ.get('MONGO_USERNAME', 'root')
DB_PASSWORD = os.environ.get('MONGO_PASSWORD', 'root')
DB_HOST = os.environ.get('MONGO_HOST', '')
DB_PORT = os.environ.get('MONGO_PORT', '')
CONN_STR = "mongodb://%s:%s@%s:%s" % (DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT)
DB_DATABASE_NAME = os.environ.get('MONGO_INITDB_DATABASE', '')
DB_MODEL_COLLECTION_NAME = os.environ.get('MONGO_MODEL_COLLECTION_NAME', '')
DB_DATA_COLLECTION_NAME = os.environ.get('MONGO_DATA_COLLECTION_NAME', '')

# 기타 변수 초기화
last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_user_input_features():
    """
    메뉴 및 사용 모델 input 값을 받기 위한 함수

    :return: json 형식의 user input 데이터
    """
    # db 에서 만들어져있는 모델명 가져오기
    client = pymongo.MongoClient(CONN_STR)
    database = client[DB_DATABASE_NAME]
    collection = database[DB_MODEL_COLLECTION_NAME]
    model_list = collection.distinct('model_name')

    user_features = {"menu_name": st.sidebar.selectbox('Menu', ['predict']),
                     "model_name": st.sidebar.selectbox('Model name', model_list)}

    # 모델 업데이트 시간 가져오기
    cursor = collection.find({'model_name': user_features["model_name"]}, {'created_timestamp': 1}).sort("created_timestamp", -1).limit(1)
    model_objects = cursor.next()
    model_created_timestamp = model_objects['created_timestamp']

    return [user_features], model_created_timestamp


def get_raw_input_features():
    """
    raw data input 값을 받기 위한 함수

    :return: json 형식의 raw input 데이터
    """
    raw_features = {"Pclass": st.sidebar.selectbox('Ticket class', [1, 2, 3]),
                    "Sex": st.sidebar.selectbox('Sex', ['male', 'female']),
                    "Age": st.sidebar.slider('Age in years', 1, 100),
                    "SibSp": st.sidebar.slider('# of siblings / spouses aboard the Titanic', 0, 10),
                    "Parch": st.sidebar.slider('# of parents / children aboard the Titanic', 0, 10),
                    "Fare": st.sidebar.slider('Passenger fare', 0, 550),
                    "Embarked": st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])
                    }

    return [raw_features]


def load_session_variable_from_model():
    """
    모델 종류가 바뀌거나 모델이 업데이트 되었을 때 새로 session variable 에 모델 정보 업데이트를 위한 함수

    :return: None
    """
    client = pymongo.MongoClient(CONN_STR)
    database = client[DB_DATABASE_NAME]
    collection = database[DB_MODEL_COLLECTION_NAME]

    cursor = collection.find({'model_name': st.session_state.model_name}).sort("created_timestamp", -1).limit(1)
    model_objects = cursor.next()

    st.session_state.created_timestamp = model_objects['created_timestamp']
    st.session_state.target_dict = pickle.loads(model_objects['target_dict'])
    st.session_state.features_selected = pickle.loads(model_objects['features_selected'])
    st.session_state.explainer = pickle.loads(model_objects['explainer'])
    st.session_state.eval_results = pickle.loads(model_objects['eval_results'])


def explain_model_prediction(data, shap_explainer, index):
    """
    shap values 및 force plot 산출을 위한 함수

    :param data: 예측 결과값을 제외하고 사용된 변수 값만 포함하는 데이터
    :param shap_explainer: 예측값의 feature 별 영향도를 파악하기 위한 shap explainer
    :param index: 예측값에 따른 shap value 선택을 위한 index
    :return: shap explainer 로 계산된 shap force plot 및 shap values
    """
    shap_values = shap_explainer.shap_values(data)
    p = shap.force_plot(shap_explainer.expected_value[index], shap_values[index], data)

    return p, shap_values


def draw_shap_plot(plot, height=None):
    """
    shap plot 을 streamlit 어플리케이션 상에 표시하기 위한 함수

    :param plot: shap force plot
    :param height: 그림 height
    :return: None
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def streamlit_main():
    """
    streamlit main 함수

    :return: None
    """
    st.title('Python ML Scoring API + Model Explainer')
    # 화면 오른쪽에 last updated 표시
    components.html(
        f'''<p style="text-align:right; font-family:'IBM Plex Sans', sans-serif; font-size:0.8rem; color:#585858";>\
            Last Updated: {last_updated}</p>''', height=30)

    # sidebar input 값 선택 UI 생성
    st.sidebar.header('User Input Features')
    user_input_data, model_created_timestamp = get_user_input_features()

    if 'model_import_flag' not in st.session_state:
        st.session_state.model_import_flag = 1
        st.session_state.model_name = user_input_data[0]['model_name']
        load_session_variable_from_model()

    if st.session_state.model_name != user_input_data[0]['model_name']:
        st.session_state.model_name = user_input_data[0]['model_name']
        load_session_variable_from_model()
    elif st.session_state.created_timestamp != model_created_timestamp:
        load_session_variable_from_model()
    else:
        pass

    st.sidebar.header('Raw Input Features')
    raw_input_data = get_raw_input_features()
    raw_input_data[0]['model_name'] = st.session_state.model_name

    submit = st.sidebar.button('Get predictions')
    if submit:
        results = requests.post(url + endpoint, json=raw_input_data)
        results = json.loads(results.text)

        # expander 형식으로 model performance 표시
        model_performance_expander = st.beta_expander('Model Performance')
        model_performance_expander.write('Accuracy: ')
        model_performance_expander.text(np.around(np.mean(st.session_state.eval_results['accuracy']), 3))
        model_performance_expander.write('Precision: ')
        model_performance_expander.text(np.around(np.mean(st.session_state.eval_results['precision']), 3))
        model_performance_expander.write('Recall: ')
        model_performance_expander.text(np.around(np.mean(st.session_state.eval_results['recall']), 3))
        model_performance_expander.write('F1_score: ')
        model_performance_expander.text(np.around(np.mean(st.session_state.eval_results['f1_score']), 3))
        model_performance_expander.write('ROC_AUC_score: ')
        model_performance_expander.text(np.around(np.mean(st.session_state.eval_results['roc_auc_score']), 3))
        model_performance_expander.write('Log Loss: ')
        model_performance_expander.text(np.around(np.mean(st.session_state.eval_results['log_loss']), 3))

        # expander 형식으로 model input 표시
        st.header('Input')
        model_input_expander = st.beta_expander('Model Input')
        model_input_expander.write('Input Features: ')
        model_input_expander.text(", ".join(list(raw_input_data[0].keys())))
        model_input_expander.json(raw_input_data[0])
        model_input_expander.write('Selected Features: ')
        model_input_expander.text(", ".join(st.session_state.features_selected))
        selected_features_values = OrderedDict((k, results[k]) for k in st.session_state.features_selected)
        model_input_expander.json(selected_features_values)

        # 예측 결과 표시
        st.header('Final Result')
        prediction = results["prediction"]
        prediction_name = [key for key in st.session_state.target_dict if st.session_state.target_dict[key] == prediction][0]
        probability = results["probability"]
        st.write("Prediction: ", int(prediction), "(", prediction_name, ")")
        st.write("Probability: ", round(float(probability), 3))

        # shap 값 계산
        results = pd.DataFrame([results])
        results.drop(['prediction', 'probability'], axis=1, inplace=True)
        results = results[st.session_state.features_selected]
        p, shap_values = explain_model_prediction(results, st.session_state.explainer, prediction)

        # shap force plot 표시
        st.subheader('Model Prediction Interpretation Plot - Label:%s' % prediction)
        draw_shap_plot(p)

        # shap feature importance plot 표시
        st.subheader('Shap Feature Importance (Absolute Value) - Label:%s' % prediction)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.summary_plot(shap_values[prediction], results, plot_type='bar')
        st.pyplot(fig)

        # expander 형식으로 shap detail 값 표시
        shap_detail_expander = st.beta_expander('Shap Detail - Label:%s' % prediction)
        for key, item in zip(st.session_state.features_selected, shap_values[prediction][0]):
            shap_detail_expander.text('%s: %s' % (key, item))


if __name__ == '__main__':
    streamlit_main()

