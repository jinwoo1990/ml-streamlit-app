import shap
from loggers import logger


def generate_shap_explainer(fitted_model):
    """
    shap 값을 계산하기 위한 explainer 와 train data 의 shap 값을 반환하는 함수

    :param fitted_model: 학습된 모델 객체
    :return: shap explainer, train 데이터 shap values
    """
    logger.info('## Generate shap explainer')
    explainer = shap.TreeExplainer(fitted_model)

    return explainer
