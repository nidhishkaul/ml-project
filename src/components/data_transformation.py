import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation for diff types of data
        '''
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(steps=[
                ('num_imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info('Numerical columns standard scaling completed')

            cat_pipeline = Pipeline(steps=[
                ('cat_imputer',SimpleImputer(strategy='most_frequent')),
                ('OHE',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info('Categorical columns encoding completed')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline, numerical_columns),
                    ('cat_pipeline',cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function is responsible for initiating data transformation process.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data.')

            preprocessor_obj = self.get_data_transformer_obj()
            
            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing data.')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Preprocessing completed.')

            # Save the preprocessor object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)