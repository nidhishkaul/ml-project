import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRanker
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data.')
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'XGBRegressor': XGBRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'LinearRegression': LinearRegression(),
                'AdaBoostClassifier': AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(x_train, y_train, x_test, y_test , models)

            ## To get best model score 
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found', sys)
            
            logging.info('Best model found')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r_score = r2_score(y_test, predicted)

            return r_score

        except Exception as e:
            raise CustomException(e, sys)
