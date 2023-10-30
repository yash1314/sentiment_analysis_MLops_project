import os
import sys

import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB


from ..exception import CustomException
from ..logger import logging

from ..utils import save_obj
from ..utils import evaluate_model   

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path =  os.path.join('artifact', 'predication_model')
    tfif_model_file_path = os.path.join('artifact', 'tfidf_model')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info('Splitting ')

            models= MultinomialNB(alpha= 20)
                
            model, score, tfidf_model  = evaluate_model(X_train,y_train, X_test, y_test, models)
            print("\n====================================================================================")
            logging.info(f'{models} : {score}')

            print(f"Model Name :{models}, Precision: {score}")
            print("\n====================================================================================")
            logging.info(f"Model Name :{models}, Precision: {score}")
            
            save_obj(
            file_path = self.model_trainer_config.trained_model_file_path,
            obj = model
            )

            save_obj(
            file_path = self.model_trainer_config.tfif_model_file_path,
            obj = tfidf_model
            )

        except Exception as e:
            logging.info('Exception occured at model trianing')
            raise CustomException(e,sys)
        