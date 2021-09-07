import pickle
import numpy as np
from utils.data_handling import load_data, save_model_to_db
from utils.explainer import generate_shap_explainer
from preprocessing.preprocessing import label_target, impute_null_features, label_features, scale_features, select_features
from classification.param_search import optimize_hyperparameters_with_random_search
from classification.evaluation import evaluate_model
from classification.train import train_model
from loggers import logger
import traceback


def train_new_model(source_type, target_col, model_name, n_folds, iteration_num, eval_metric, random_state=42,
                    file_flag=0, input_path='', output_path=''):
    # 데이터 불러오기
    data = load_data(source_type=source_type, input_path=input_path)

    # 타겟 컬럼 labeling
    data, target_dict = label_target(data=data, target_col=target_col)

    # 필요없는 컬럼 제거
    # TODO: 데이터마다 바뀜
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore', inplace=True)

    # null 값 impute
    data, null_converter = impute_null_features(data=data)

    # 파생변수 생성
    # TODO: 데이터마다 바뀜
    logger.info('## Create derived variables')
    data["Fsize"] = data["SibSp"] + data["Parch"] + 1
    # 파생변수 생성 후 필요없는 컬럼 제거
    data = data.drop(["SibSp", "Parch"], axis=1)
    cols = list(data.columns.difference(['target'])) + ['target']
    data = data[cols]
    logger.info('data: \n %s' % list(data.columns.difference(['target'])))

    # categorical features 파악
    # TODO: 데이터에 따라 직접 지정
    categorical_feats_name = [col for col in data.columns if data[col].dtype == 'object']

    # feature 컬럼 labeling
    # lightgbm 이 categorical feats 도 int, float 형식으로 다 바꿔야 작동하므로 일괄 적용
    data, label_encoder = label_features(data)

    # scaling
    if model_name in ['rf', 'xgb', 'lgb', 'catboost']:
        pass
    else:
        data = scale_features(data)

    # feature selection
    features_selected = select_features(data=data, categorical_feats_name=categorical_feats_name)

    # categorical feats index 찾기
    cols = features_selected + ['target']
    data = data[cols]
    categorical_feats = [idx for idx, col in enumerate(data.columns) if col in categorical_feats_name]
    logger.info('Categorical feats idx: \n %s' % categorical_feats)
    logger.info('Categorical feats name: \n %s' % categorical_feats_name)
    logger.info('data: \n %s' % data.iloc[:, categorical_feats])

    # X, y 분리
    # 여기부터 numpy
    X = data[features_selected].values
    y = data['target'].values

    # hyper parameters search
    optimized_params = optimize_hyperparameters_with_random_search(X=X, y=y, model_name=model_name,
                                                                   n_folds=n_folds, iteration_num=iteration_num,
                                                                   eval_metric=eval_metric,
                                                                   categorical_feats=categorical_feats,
                                                                   random_state=random_state)

    # 모델 evaluation
    eval_results = evaluate_model(X=X, y=y, model_name=model_name, optimized_params=optimized_params,
                                  n_folds=n_folds, eval_metric=eval_metric,
                                  categorical_feats=categorical_feats,
                                  random_state=random_state)

    # 모델 training
    fitted_model = train_model(X=X, y=y, model_name=model_name, optimized_params=optimized_params,
                               eval_metric=eval_metric, categorical_feats=categorical_feats)

    # shap explainer 생성
    explainer = generate_shap_explainer(fitted_model=fitted_model)

    # 저장
    target_dict_pkl = pickle.dumps(target_dict)
    null_converter_pkl = pickle.dumps(null_converter)
    label_encoder_pkl = pickle.dumps(label_encoder)
    features_selected_pkl = pickle.dumps(features_selected)
    optimized_params_pkl = pickle.dumps(optimized_params)
    fitted_model_pkl = pickle.dumps(fitted_model)
    eval_results_pkl = pickle.dumps(eval_results)
    explainer_pkl = pickle.dumps(explainer)

    model_objects = {
        'model_name': model_name,
        'target_dict': target_dict_pkl,
        'null_converter': null_converter_pkl,
        'label_encoder': label_encoder_pkl,
        'features_selected': features_selected_pkl,
        'optimized_params': optimized_params_pkl,
        'fitted_model': fitted_model_pkl,
        'eval_results': eval_results_pkl,
        'explainer': explainer_pkl
    }

    if file_flag == 1:
        with open(output_path, 'wb') as handle:
            pickle.dump(model_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        save_model_to_db(model_objects)


def transform_record(data, model_name, null_converter, label_encoder, features_selected):
    """
    categorical 컬럼의 label encoding 결과를 반환하는 함수

    :param data: 전처리되지 않은 input 데이터
    :param model_name: 모델 종류
    :param null_converter: null 값을 처리하기 위한 categorical column 최빈값 dictionary
    :param label_encoder: categorical 값 변환을 위한 label encoder dictionary (sklearn.preprocessing.OrdinalEncoder)
    :param features_selected: feature selection 과정에서 선택된 features list
    :return: label encoding 이 완료되고 모델링에 필요한 features 만 선택된 dataframe
    """
    # 필요없는 컬럼 제거
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore', inplace=True)

    # null 값 training 데이터 최빈값으로 변환
    for col in null_converter.keys():
        temp = data[col].fillna(null_converter[col])
        data[col] = temp

    # 파생변수 생성
    data["Fsize"] = data["SibSp"] + data["Parch"] + 1
    data = data.drop(["SibSp", "Parch"], axis=1)

    # 컬럼별로 정의된 OrdinalEncoder 를 사용해 categorical 데이터 변환 (처음 나오는 값은 -1로 변환됨)
    for col in list(label_encoder.keys()):
        temp = label_encoder[col].transform(np.array(data[col]).reshape(-1, 1))
        data[col] = temp

    return data[features_selected]


def predict_record(data, model):
    """
    예측값과 예측확률을 반환하는 함수

    :param data: transform_record() 을 통해 전치리가 된 input 데이터
    :param model: 예측 모델
    :return: 모델 예측값 및 예측확률
    """
    prediction = model.predict(data)[0]
    probability = np.max(model.predict_proba(data))

    return prediction, probability


if __name__ == '__main__':
    try:
        # db
        train_new_model(source_type='db', target_col='Survived', model_name='rf',
                        n_folds=5, iteration_num=5, eval_metric='log_loss')

        # csv
        # train_new_model(source_type='csv', target_col='Survived', model_name='rf',
        #                 n_folds=5, iteration_num=5, eval_metric='log_loss',
        #                 file_flag=1, input_path='./train.csv', output_path='./model_objects.pkl')

    except Exception:
        logger.error(traceback.format_exc())
        raise
