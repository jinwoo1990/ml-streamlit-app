import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from classification.base import fit_rf, fit_lgb
from loggers import logger


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
            fitted_model = fit_rf(X=X, y=y, optimized_params=optimized_params)
        elif model_name == 'lgb':
            fitted_model = fit_lgb(X=X, y=y, optimized_params=optimized_params,
                                   eval_metric=eval_metric, categorical_feats=categorical_feats)
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
