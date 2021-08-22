from collections import defaultdict
import os
import time
import random
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFECV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
import shap
import helper
import traceback
from base_logger import logger


DB_DATABASE_NAME = os.environ.get('MONGO_INITDB_DATABASE', 'ml')
DB_MODEL_COLLECTION_NAME = os.environ.get('MONGO_MODEL_COLLECTION_NAME', 'model')
DB_DATA_COLLECTION_NAME = os.environ.get('MONGO_DATA_COLLECTION_NAME', 'data')


def load_data(input_path='', source_type='db'):
    """
    데이터 로드를 위한 함수

    :param input_path: input 경로
    :param source_type: source 데이터 유형
    :return: input 경로에서 읽어온 raw 데이터
    """
    logger.info('## Load data')
    if source_type == 'db':
        raw_data = helper.load_data_from_db(DB_DATABASE_NAME, DB_DATA_COLLECTION_NAME)
        raw_data = raw_data.replace('', np.nan, regex=True)
    elif source_type == 'csv':
        raw_data = pd.read_csv(input_path, sep=',')
    else:
        logger.error("source_type not recognized: should be 'db' or 'csv'")
        raise ValueError

    logger.info('raw_data: \n %s' % raw_data.head())

    return raw_data


def preprocess_data(raw_data, target_col, scaling_flag):
    """
    전처리 결과를 반환하는 함수

    :param raw_data: target labeling 이 된 데이터
    :param target_col: target 컬럼
    :param scaling_flag: scaling 적용 여부 flag
    :return: 전처리가 된 train 데이터, null 값 처리를 위한 최빈값 dictionary, categorical labeling 을 위한 label dictionary
    """
    logger.info('## Preprocess data')
    # target 값 labeling
    target_label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    target_label_encoder.fit(np.array(raw_data[target_col]).reshape(-1, 1))
    target_dict = {}
    for idx, item in enumerate(target_label_encoder.categories_[0]):
        target_dict[item] = idx

    raw_data['target'] = raw_data[target_col].apply(lambda x: target_dict[x])
    labeled_data = raw_data.drop(target_col, axis=1)

    # 필요없는 컬럼 제거
    labeled_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore', inplace=True)
    logger.info('labeled_data: \n %s' % labeled_data)

    # 결측값 제거
    # null 값 변환하기 위한 각 컬럼별 최빈값 산출
    null_counts = labeled_data.isnull().sum()
    null_cols = list(null_counts[null_counts > 0].index)
    logger.info('null_cols: \n %s' % null_cols)
    null_converter = defaultdict(str)
    for col in null_cols:
        null_converter[col] = labeled_data[col].value_counts().index[0]
    filled_data = labeled_data.copy()
    for col in null_converter.keys():
        temp = filled_data[col].fillna(null_converter[col])
        filled_data[col] = temp

    # categorical features 처리
    preprocessed_data = filled_data.copy()
    # 경우에 따라 일부 data 는 dummy 로 변환해야 할 수 있음
    # 컬럼별 ordinal label encoder 생성 (ordinal 하게 해석될 수 있는 컬럼 대상)
    object_cols = preprocessed_data.select_dtypes(include=['object']).columns
    label_encoder = {}
    for col in object_cols:
        label_encoder[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        label_encoder[col].fit(np.array(preprocessed_data[col]).reshape(-1, 1))
    # 데이터 변환
    for col in label_encoder.keys():
        temp = label_encoder[col].transform(np.array(preprocessed_data[col]).reshape(-1, 1))
        preprocessed_data[col] = temp

    # 파생변수 생성
    preprocessed_data["Fsize"] = preprocessed_data["SibSp"] + preprocessed_data["Parch"] + 1
    # 파생변수 생성 후 필요없는 컬럼 제거
    preprocessed_data = preprocessed_data.drop(["SibSp", "Parch"], axis=1)

    # scaling
    if scaling_flag == 1:
        X = preprocessed_data[preprocessed_data.columns.difference(['target'])]
        scaler = StandardScaler()
        preprocessed_data[preprocessed_data.columns.difference(['target'])] = scaler.fit_transform(X)

    logger.info('preprocessed_data: \n %s' % preprocessed_data)
    # print(preprocessed_data)

    return preprocessed_data, target_dict, null_converter, label_encoder


def select_features(X, y):
    """
    features selection 결과를 반환하는 함수

    :param X: target 이 제외된 데이터
    :param y: target 데이터
    :return: 중요 features list
    """
    logger.info('## Select features')
    logger.info('Original feature numbers: %s' % len(X.columns))

    # 해석 용이성을 위한 multicollinearity 제거
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['Feature'] = X.columns
    logger.info('VIF: \n %s' % vif)
    # VIF > 10 제거
    deleted_mask = [variance_inflation_factor(X.values, i) > 10 for i in range(X.shape[1])]
    features_with_multicollinearity = list(X.columns[deleted_mask])
    X = X.drop(features_with_multicollinearity, axis=1)
    logger.info('X: \n %s' % X)

    estimator = RandomForestClassifier(random_state=20, n_estimators=100, n_jobs=-1)
    selector = RFECV(estimator, step=1, cv=5, min_features_to_select=5, n_jobs=-1)
    selector = selector.fit(X, y)

    logger.info('%s features selected' % selector.n_features_)
    selected_mask = selector.support_
    features_selected = list(X.columns[selected_mask])
    logger.info('Selected features: \n %s' % features_selected)

    return features_selected


def optimize_hyperparameters_with_random_search(X, y, model_name, n_folds, iteration_num,
                                                eval_metric='log_loss', categorical_feats=None, random_state=42):
    """
    최적화된 hyperparameters 를 random search 를 통해 찾는 함수

    :param X: target 이 제외된 데이터
    :param y: target 데이터
    :param model_name: 모델 종류
    :param n_folds: hyperparameter search 및 evaluation 에 사용할 fold 갯수
    :param iteration_num: hyperparameter search 반복 횟수
    :param eval_metric: hyperparameter search 및 evaluation 에 사용할 evaluation metric
    :param categorical_feats: lightGBM 등 categorical features 를 자동 처리해주는 알고리즘에 넘겨주기 위한 features 목록
    :param random_state: random state
    :return: 최적화된 parameters
    """
    logger.info('## Optimize hyperparameters')
    lowest_cv = 9999999
    optimized_params = {}

    start = time.time()

    for i in range(iteration_num):
        logger.info('For {} of {} iterations'.format(i + 1, iteration_num))

        if model_name == 'rf':
            # mongoDB 사용 시 16 mb 용량 한계로 parameter 범위 작게 잡음
            params = {'n_estimators': np.random.randint(10, 30),
                      'max_features': random.choice(['auto', 'sqrt', 'log2']),
                      'max_depth': np.random.randint(5, 10),
                      'min_samples_split': random.choice([2, 5, 10]),
                      'min_samples_leaf': random.choice([1, 2, 4]),
                      'bootstrap': random.choice([True, False])}

            kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

            eval_results = {eval_metric: []}

            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                rf = RandomForestClassifier(**params)
                rf.fit(X_train, y_train)

                y_true = y_test
                y_pred = rf.predict(X_test)

                fold_log_loss = log_loss(y_true, y_pred)
                eval_results[eval_metric].append(fold_log_loss)

            min_cv_results = np.mean(eval_results[eval_metric])

            if min_cv_results < lowest_cv:
                logger.info('Params changed')
                lowest_cv = min_cv_results
                optimized_params = params

        elif model_name == 'lgb':
            if eval_metric == 'log_loss':
                eval_metric_str = 'binary_logloss'
            else:
                logger.error("eval_metric not recognized: should be 'log_loss' or ...")
                raise ValueError

            params = {'objective': 'binary',
                      'metric': eval_metric_str,
                      'num_leaves': np.random.randint(24, 48),
                      'max_depth': np.random.randint(5, 8),
                      'min_child_weight': np.random.randint(5, 50),
                      'min_split_gain': np.random.rand() * 0.09,
                      'colsample_bytree': np.random.rand() * (0.9 - 0.1) + 0.1,
                      'subsample': np.random.rand() * (1 - 0.8) + 0.8,
                      'bagging_freq': np.random.randint(1, 5),
                      'bagging_seed': np.random.randint(1, 5),
                      'reg_alpha': np.random.rand() * 2,
                      'reg_lambda': np.random.rand() * 2,
                      'learning_rate': np.random.rand() * 0.02,
                      'seed': 1989,
                      'verbose': -1,
                      'num_threads': 1}

            d_train = lgb.Dataset(data=X, label=y, categorical_feature=categorical_feats,
                                  free_raw_data=False)
            cv_results = lgb.cv(params=params, train_set=d_train, num_boost_round=1000,
                                categorical_feature=categorical_feats,
                                nfold=n_folds, stratified=True, shuffle=True, early_stopping_rounds=10,
                                verbose_eval=1000)

            min_cv_results = min(cv_results['binary_logloss-mean'])

            if min_cv_results < lowest_cv:
                logger.info('Params changed')
                lowest_cv = min_cv_results
                optimized_params = params
        else:
            logger.error("model_type not recognized: should be 'rf' or 'lgb'")
            raise ValueError

    end = time.time()

    logger.info('Elapsed time for optimization: %s' % (end - start))

    return optimized_params


def evaluate_model(X, y, model_name, optimized_params, n_folds, eval_metric='log_loss', categorical_feats=None, random_state=42):
    """
    모델 평가를 위한 함수

    :param X: target 이 제외된 데이터
    :param y: target 데이터
    :param model_name: 모델 종류
    :param optimized_params: tuning 된 모델 파라미터
    :param n_folds: hyperparameter search 및 evaluation 에 사용할 fold 갯수
    :param eval_metric: hyperparameter search 및 evaluation 에 사용할 evaluation metric
    :param categorical_feats: lightGBM 등 categorical features 를 자동 처리해주는 알고리즘에 넘겨주기 위한 features 목록
    :param random_state: random state
    :return: 모델 평가 결과
    """
    logger.info('## Evaluate model')
    # StratifiedKFold 로 안 하면 class 가 imbalanced 된 데이터의 경우 metric 측정이 제대로 안될 수 있음
    kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

    eval_results = {
        'train_time': [],
        'confusion_matrix': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc_score': [],
        'log_loss': []
    }

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        start = time.time()
        if model_name == 'rf':
            fitted_model = RandomForestClassifier(**optimized_params)
            fitted_model.fit(X, y)
        elif model_name == 'lgb':
            if eval_metric == 'log_loss':
                eval_metric_str = 'logloss'
            else:
                logger.error("eval_metric not recognized: should be 'log_loss' or ...")
                raise ValueError

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y)
            fitted_model = lgb.LGBMClassifier(**optimized_params)
            fitted_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=eval_metric_str,
                             categorical_feature=categorical_feats,
                             early_stopping_rounds=50, verbose=500)
        else:
            logger.error("model_type not recognized: should be 'rf' or 'lgb'")
            raise ValueError
        end = time.time()

        y_true = y_test
        y_pred = fitted_model.predict(X_test)

        fold_train_time = end - start
        fold_confusion_matrix = confusion_matrix(y_true, y_pred)
        fold_accuracy = accuracy_score(y_true, y_pred)
        fold_precision = precision_score(y_true, y_pred)
        fold_recall = recall_score(y_true, y_pred)
        fold_f1_score = f1_score(y_true, y_pred)
        fold_roc_auc_score = roc_auc_score(y_true, y_pred)
        fold_log_loss = log_loss(y_true, y_pred)

        eval_results['train_time'].append(fold_train_time)
        eval_results['confusion_matrix'].append(fold_confusion_matrix)
        eval_results['accuracy'].append(fold_accuracy)
        eval_results['precision'].append(fold_precision)
        eval_results['recall'].append(fold_recall)
        eval_results['f1_score'].append(fold_f1_score)
        eval_results['roc_auc_score'].append(fold_roc_auc_score)
        eval_results['log_loss'].append(fold_log_loss)

    logger.info('eval_results: \n %s' % eval_results)

    return eval_results


def train_model(X, y, model_name, optimized_params, eval_metric='log_loss', categorical_feats=None):
    """
    모델 학습을 위한 함수

    :param X: target 이 제외된 데이터
    :param y: target 데이터
    :param model_name: 모델 종류
    :param optimized_params: tuning 된 모델 파라미터
    :param eval_metric: early stopping 이용한 학습 시 evaluation metric
    :param categorical_feats: 자동 encoding 할 categorical featatures (lgbm 등에서 사용)
    :return: 학습된 모델 객체
    """
    logger.info('## Train model')
    if model_name == 'rf':
        fitted_model = RandomForestClassifier(**optimized_params)
        fitted_model.fit(X, y)
    elif model_name == 'lgb':
        if eval_metric == 'log_loss':
            eval_metric_str = 'logloss'
        else:
            logger.error("eval_metric not recognized: should be 'log_loss' or ...")
            raise ValueError

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y)
        fitted_model = lgb.LGBMClassifier(**optimized_params)
        fitted_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=eval_metric_str,
                         categorical_feature=categorical_feats,
                         early_stopping_rounds=50, verbose=500)
    else:
        logger.error("model_type not recognized: should be 'rf' or 'lgb'")
        raise ValueError

    return fitted_model


def generate_shap_explainer(fitted_model):
    """
    shap 값을 계산하기 위한 explainer 와 train data 의 shap 값을 반환하는 함수

    :param fitted_model: 학습된 모델 객체
    :return: shap explainer, train 데이터 shap values
    """
    logger.info('## Generate shap explainer')
    explainer = shap.TreeExplainer(fitted_model)

    return explainer


def create_model_objects(source_type, target_col, model_name, n_folds, iteration_num, eval_metric, input_path='', output_path='', file_flag=0):
    """
    api 에 사용할 모델 objects 를 생성하는 함수

    :param source_type: source 데이터 유형
    :param target_col: target 컬럼
    :param model_name: 모델 종류
    :param n_folds: hyperparameter search 및 evaluation 에 사용할 fold 갯수
    :param iteration_num: hyperparameter search 반복 횟수
    :param eval_metric: hyperparameter search 및 evaluation 에 사용할 evaluation metric
    :param input_path: input 경로
    :param output_path: output 경로
    :param file_flag: pickle 파일 생성 flag
    :return: None
    """
    # 데이터 불러오기
    raw_data = load_data(input_path, source_type)

    # 전처리
    if model_name in ['rf', 'xgb', 'lgb', 'catboost']:
        scaling_flag = 0
    else:
        scaling_flag = 1
    preprocessed_data, target_dict, null_converter, label_encoder = preprocess_data(raw_data, target_col, scaling_flag)

    # X, y 분리
    X = preprocessed_data[preprocessed_data.columns.difference(['target'])]
    y = preprocessed_data['target']

    # feature selection
    features_selected = select_features(X, y)
    X = X[features_selected].values
    y = y.values

    # hyper parameters search
    optimized_params = optimize_hyperparameters_with_random_search(X, y, model_name, n_folds, iteration_num, eval_metric)

    # 모델 evaluation
    eval_results = evaluate_model(X, y, model_name, optimized_params, n_folds, eval_metric)

    # 모델 training
    fitted_model = train_model(X, y, model_name, optimized_params, eval_metric)

    # shap explainer 생성
    explainer = generate_shap_explainer(fitted_model)

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
        helper.save_model_to_db(DB_DATABASE_NAME, DB_MODEL_COLLECTION_NAME, model_objects)


if __name__ == '__main__':
    try:
        create_model_objects('cv', 'Survived', 'rf', 5, 5, 'log_loss',
                             input_path='./train.csv', output_path='./model_objects.pkl', file_flag=1)
    except Exception:
        logger.error(traceback.format_exc())
        raise
