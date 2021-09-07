from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFECV
from loggers import logger


def label_target(data, target_col):
    """
    target 을 labeling 하기 위한 함수

    :param data: raw 데이터
    :param target_col: target 컬럼
    :return: target labeling 이 된 데이터, 원천 target 정보를 포함한 dictionary
    """
    logger.info('## Label target')

    target_label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    target_label_encoder.fit(np.array(data[target_col]).reshape(-1, 1))
    target_dict = {}
    for idx, item in enumerate(target_label_encoder.categories_[0]):
        target_dict[item] = idx

    data['target'] = data[target_col].apply(lambda x: target_dict[x])
    data = data.drop(target_col, axis=1)

    logger.info('data: \n %s' % data.head())

    return data, target_dict


def impute_null_features(data):
    """
    null 값을 채우기 위한 함수

    :param data: target labeling 이 된 데이터
    :return: null 값 처리를 위한 최빈값 dictionary
    """
    logger.info('## Impute null features')

    # null 값 변환하기 위한 각 컬럼별 최빈값 산출
    null_counts = data.isnull().sum()
    null_cols = list(null_counts[null_counts > 0].index)
    logger.info('null_cols: \n %s' % null_cols)
    null_converter = defaultdict(str)
    for col in null_cols:
        null_converter[col] = data[col].value_counts().index[0]
    for col in null_converter.keys():
        temp = data[col].fillna(null_converter[col])
        data[col] = temp

    logger.info('data: \n %s' % data.head())

    return data, null_converter


def label_features(data):
    """
    categorical features 의 labeling 을 위한 함수

    :param data: null 값이 채워진 데이터
    :return: label 된 데이터, categorical labeling 을 위한 label dictionary
    """
    logger.info('## Label features')

    # 경우에 따라 일부 data 는 dummy 로 변환해야 할 수 있음
    # 컬럼별 ordinal label encoder 생성 (ordinal 하게 해석될 수 있는 컬럼 대상)
    object_cols = data.select_dtypes(include=['object']).columns
    label_encoder = {}
    for col in object_cols:
        label_encoder[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        label_encoder[col].fit(np.array(data[col]).reshape(-1, 1))
    # 데이터 변환
    for col in label_encoder.keys():
        temp = label_encoder[col].transform(np.array(data[col]).reshape(-1, 1))
        data[col] = temp

    logger.info('data: \n %s' % data.head())

    return data, label_encoder


def scale_features(data):
    """
    scaling 을 위한 함수

    :param data: 기타 전처리가 끝난 데이터
    :return: scale 된 데이터
    """
    logger.info('## Scale features')

    X = data[data.columns.difference(['target'])]
    scaler = StandardScaler()
    data[data.columns.difference(['target'])] = scaler.fit_transform(X)

    logger.info('data: \n %s' % data.head())

    return data


def select_features(data, categorical_feats_name):
    """
    features selection 결과를 반환하는 함수

    :param data: preprocessed 된 데이터
    :param categorical_feats_name: categorical features 이름
    :return: 중요 features list
    """
    logger.info('## Select features')
    logger.info('Original feature numbers: %s' % len(data.columns.difference(['target'])))
    logger.info('Original features name: \n %s' % list(data.columns.difference(['target'])))
    logger.info('Categorical features name: %s' % categorical_feats_name)

    X = data[data.columns.difference(['target'] + categorical_feats_name)]
    y = data['target']

    # 해석 용이성을 위한 multicollinearity 제거
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['Feature'] = X.columns
    logger.info('VIF: \n %s' % vif)
    # VIF > 10 제거
    deleted_mask = [variance_inflation_factor(X.values, i) > 10 for i in range(X.shape[1])]
    features_with_multicollinearity = list(X.columns[deleted_mask])
    X = X.drop(features_with_multicollinearity, axis=1)

    estimator = RandomForestClassifier(random_state=20, n_estimators=100, n_jobs=-1)
    selector = RFECV(estimator, step=1, cv=5, min_features_to_select=5, n_jobs=-1)
    selector = selector.fit(X, y)

    selected_mask = selector.support_
    features_selected = list(X.columns[selected_mask]) + categorical_feats_name
    logger.info('%s features selected' % len(features_selected))
    logger.info('Selected features: \n %s' % features_selected)

    return features_selected
