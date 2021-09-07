import time
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import log_loss
from loggers import logger


def random_search_rf(lowest_cv, X, y, n_folds, eval_metric, random_state):
    """
    random forest 의 random search 를 위한 함수

    :param lowest_cv: 이전 iteration 의 lowest cv score
    :param X: target 이 제외된 데이터
    :param y: target 데이터
    :param n_folds: hyperparameter search 및 evaluation 에 사용할 fold 갯수
    :param eval_metric: hyperparameter search 및 evaluation 에 사용할 evaluation metric
    :param random_state: random state
    :return: 최적화된 parameters, 갱신된 lowest cv score
    """
    optimized_params = {}

    if eval_metric not in ['log_loss']:
        logger.error("eval_metric not recognized: should be 'log_loss' or ...")
        raise ValueError

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

    return optimized_params, lowest_cv


def random_search_lgb(lowest_cv, X, y, categorical_feats, n_folds, eval_metric):
    """
    lightgbm 의 random search 를 위한 함수

    :param lowest_cv: 이전 iteration 의 lowest cv score
    :param X: target 이 제외된 데이터
    :param y: target 데이터
    :param categorical_feats: lightGBM 등 categorical features 를 자동 처리해주는 알고리즘에 넘겨주기 위한 features 목록
    :param n_folds: hyperparameter search 및 evaluation 에 사용할 fold 갯수
    :param eval_metric: hyperparameter search 및 evaluation 에 사용할 evaluation metric
    :return: 최적화된 parameters, 갱신된 lowest cv score
    """
    optimized_params = {}

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

    d_train = lgb.Dataset(data=X, label=y,
                          free_raw_data=False)
    cv_results = lgb.cv(params=params, train_set=d_train, num_boost_round=1000,
                        nfold=n_folds, stratified=True, shuffle=True, early_stopping_rounds=10,
                        verbose_eval=1000)

    min_cv_results = min(cv_results['binary_logloss-mean'])

    if min_cv_results < lowest_cv:
        logger.info('Params changed')
        lowest_cv = min_cv_results
        optimized_params = params

    return optimized_params, lowest_cv


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
            optimized_params, lowest_cv = random_search_rf(lowest_cv=lowest_cv, X=X, y=y,
                                                           n_folds=n_folds, eval_metric=eval_metric,
                                                           random_state=random_state)
        elif model_name == 'lgb':
            optimized_params, lowest_cv = random_search_lgb(lowest_cv=lowest_cv, X=X, y=y,
                                                            categorical_feats=categorical_feats,
                                                            n_folds=n_folds, eval_metric=eval_metric)
        else:
            logger.error("model_type not recognized: should be 'rf' or 'lgb'")
            raise ValueError

    end = time.time()

    logger.info('Elapsed time for optimization: %s' % (end - start))

    return optimized_params
