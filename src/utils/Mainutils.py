import sys,os,pickle,yaml
from typing import Dict,Tuple
import pandas as pd
import numpy as np
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from collections import defaultdict

class MainUtils:
    def __init__(self) -> None:
        pass


    @staticmethod
    def save_object(file_path,obj):
        logging.info("Entered the save_object method of MainUtils class")
        try:
            dir_path=os.path.dirname(file_path)
            os.makedirs(dir_path,exist_ok=True)

            with open(file_path,'wb') as file_obj:
                pickle.dump(obj,file_obj)
        except Exception as e:
            logging.info("error occured during saving!")

    @staticmethod
    def load_object(file_path:str)->object:
        logging.info('Entered the load_object method of mainutils class')
        try:
            with open(file_path,'rb') as file_obj:
                return pickle.load(file_obj)
            
        except Exception as e:
             logging.info("error occured during loading!")
             raise CustomException(e,sys)
    
    @staticmethod
    def get_error_metrics(y_true, y_pred):
        rmse = np.sqrt(np.mean([ (y_true[i] - y_pred[i])**2 for i in range(len(y_pred)) ]))
        mape = np.mean(np.abs( (y_true - y_pred)/y_true )) * 100
        return rmse, mape
    
    @staticmethod
    def get_rmse_metrics(y_true, y_pred):
        rmse = np.sqrt(np.mean([ (y_true[i] - y_pred[i])**2 for i in range(len(y_pred)) ]))
        return rmse
    
    @staticmethod
    def precision_recall_at_k(y, y_pred, user_list, k=10, threshold = 7):
        """Return precision and recall at k metrics for each user"""

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for i in range(len(y)):
            user_est_true[user_list[i]].append((y_pred[i], y[i]))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            
            # Number of relevant items
            n_rel = np.sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = np.sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = np.sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls