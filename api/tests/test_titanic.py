import unittest
import os
import pandas as pd
import pymongo
from pipeline_titanic import train_new_model
from loggers import logger


DB_USERNAME = os.environ.get('MONGO_USERNAME', 'root')
DB_PASSWORD = os.environ.get('MONGO_PASSWORD', 'root')
DB_HOST = os.environ.get('MONGO_HOST', '')
DB_PORT = os.environ.get('MONGO_PORT', '')
CONN_STR = "mongodb://%s:%s@%s:%s" % (DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT)
DB_DATABASE_NAME = os.environ.get('MONGO_INITDB_DATABASE', 'ml')
DB_MODEL_COLLECTION_NAME = os.environ.get('MONGO_MODEL_COLLECTION_NAME', 'model')
DB_DATA_COLLECTION_NAME = os.environ.get('MONGO_DATA_COLLECTION_NAME', 'data')


class MyTests(unittest.TestCase):
    def setUp(self):
        self.client = pymongo.MongoClient(CONN_STR)
        self.database = self.client[DB_DATABASE_NAME]

    def tearDown(self):
        del self.client
        del self.database

    def test_load_data_from_db(self):
        print('')
        collection = self.database[DB_DATA_COLLECTION_NAME]
        cursor = collection.find()
        result = list(cursor)
        df = pd.DataFrame(result)
        del df['_id']
        logger.info('loaded_data: \n %s' % df.head())

        df_test = pd.read_csv('train.csv')

        self.assertEqual(df.shape, df_test.shape)
        self.assertEqual(list(df.columns), list(df_test.columns))

    def test_create_model_objects(self):
        print('')
        collection = self.database[DB_MODEL_COLLECTION_NAME]
        model_count = collection.count_documents({'model_name': 'rf'})

        train_new_model('db', 'Survived', 'rf', 2, 2, 'log_loss')
        model_count_test = collection.count_documents({'model_name': 'rf'})

        self.assertEqual(model_count, model_count_test - 1)

        item = collection.find({'model_name': 'rf'}).sort("created_timestamp", -1).limit(1).next()
        collection.delete_one({'_id': item['_id']})


if __name__ == '__main__':
    unittest.main()
