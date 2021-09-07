from classification.base import fit_rf, fit_lgb
from loggers import logger


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
        fitted_model = fit_rf(X=X, y=y, optimized_params=optimized_params)
    elif model_name == 'lgb':
        fitted_model = fit_lgb(X=X, y=y, optimized_params=optimized_params,
                               eval_metric=eval_metric, categorical_feats=categorical_feats)
    else:
        logger.error("model_type not recognized: should be 'rf' or 'lgb'")
        raise ValueError

    return fitted_model
