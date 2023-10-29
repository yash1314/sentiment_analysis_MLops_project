import sys
import os
from ..exception import CustomException
from ..utils import load_object, nlp_transform
from ..logger import logging

import pandas as pd
import numpy as np

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try:
            ## model loading 
            model_path = os.path.join('artifact', 'predication_model')
            tfidf_path = os.path.join('artifact', 'tfidf_model')
            
            ## model object creation 
            model = load_object(model_path)
            tfidf = load_object(tfidf_path)

            trf_features = pd.DataFrame([{'lemma_transform': f'{features}'}])

            ## nlp transformation
            final_features = trf_features['lemma_transform'].apply(nlp_transform)

            ## text Vectorization
            features_vec = tfidf.transform(final_features)

            ## prediction
            pred = model.predict(features_vec)
            return pred
        
        except Exception as e:
            logging.info('Error occured in predict pipeline folder')
            raise CustomException(e, sys)
        