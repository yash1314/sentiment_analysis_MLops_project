import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 


import numpy as np
import pandas as pd

from ..exception import CustomException
from ..logger import logging
from ..utils import * 

from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    transformed_data_file_path = os.path.join('artifact', 'transformed_data.csv')
    

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


   
    def initiate_data_transformation(self, data_path):
        try : 
            ## reading the data
            df = pd.read_csv(data_path)

            #dropping duplicate 
            df.drop_duplicates(keep='first', inplace=True)

            # removing label values tagged as 'O'
            condition = df['label'] == 'O'
            df = df[~condition]

            # encoding label values
            df['encoded_label'] = df['label'].replace({'N': 0, 'P': 1})

            # NLP text conversion
            df['lemma_transform'] = df['comment'].apply(nlp_transform)

            # final columns drop
            df.drop(['Unnamed: 0','comment', 'label'], axis = 1, inplace = True)
            
            logging.info('df data transformation completed')
            logging.info(f' transformed df data head: \n{df.head().to_string()}')

            df.to_csv(self.data_transformation_config.transformed_data_file_path, index = False, header= True)
            logging.info("transformed data is stored")
            df.head(1)

            #using a subset of data for model making
            i = 17500
            positive_samples = df[df['encoded_label'] == 1]
            negative_samples = df[df['encoded_label'] == 0]

            # Perform random subsampling for each class
            # You can adjust the subsample size for each class as needed
            positive_subsample = positive_samples.sample(n = i, random_state = 0)
            negative_subsample = negative_samples.sample(n = i, random_state = 0)

            # Combine the subsamples into a single DataFrame
            subsampled_df = pd.concat([positive_subsample, negative_subsample])

            # Shuffle the entire DataFrame to randomize the order of samples
            subsampled_df = shuffle(subsampled_df, random_state=0).reset_index(drop=True)


            ## splitting the data into training and target data
            X = subsampled_df['lemma_transform']
            y = subsampled_df['encoded_label']
            
            ## further Splitting the data.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, shuffle = True) 
            logging.info('final splitting the data is successful')
            
            ## returning splitted data and data_path.
            return (
                X_train, 
                X_test, 
                y_train, 
                y_test,
                self.data_transformation_config.transformed_data_file_path
            )
        
        except Exception as e:
            logging.info('error occured in the initiate_data_transformation')
            raise CustomException(e, sys)
