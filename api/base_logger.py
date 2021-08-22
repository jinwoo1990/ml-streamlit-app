import logging.config
import json
import os


# os.makedirs('./logs', exist_ok=True)
# 그냥 WORKDIR 에 logger_config 넣고 쓰자
LOG_CONFIG_PATH = 'logger_config.json'

logging.config.dictConfig(json.load(open(LOG_CONFIG_PATH)))
logger = logging.getLogger('base')

logger.debug('********** logger initiated')
