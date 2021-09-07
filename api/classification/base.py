from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from loggers import logger


def fit_rf(X, y, optimized_params):
    fitted_model = RandomForestClassifier(**optimized_params)
    fitted_model.fit(X, y)

    return fitted_model


def fit_lgb(X, y, optimized_params, eval_metric, categorical_feats):
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

    return fitted_model
