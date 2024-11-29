import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils.Mainutils import MainUtils
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from scipy.sparse import hstack
import pickle
import tqdm
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from src.components.content_based import Content_Based_Reccomendation

import surprise as sp
from surprise import accuracy
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.knns import KNNBaseline
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms.matrix_factorization import NMF

@dataclass
class Hybrid_Reccomendation_config:
    # knnbaseline predictions
    knnbaseline_pred_pth=os.path.join('artifacts','knn_baseline_pred_dict.pkl')


class Hybrid_Reccomendation:
    def __init__(self):
        self.utils=MainUtils()
        self.data_ing=DataIngestion()
        self.data_trf=DataTransformation()
        self.content_based=Content_Based_Reccomendation()
        self.hybrid_config=Hybrid_Reccomendation_config()
    
    def train_collaborative(self):
        try:
            logging.info('initializing training')
            df_train_sample=pd.read_csv(self.data_ing.data_ingestion_config.train_data_path)
            df_test_sample=pd.read_csv(self.data_ing.data_ingestion_config.test_data_path)
            
            reader = sp.reader.Reader(rating_scale = (1, 10))
            train_data = sp.Dataset.load_from_df(df_train_sample[['user_id', 'anime_id', 'my_score']], reader)

            trainset = train_data.build_full_trainset()
            testset = list(zip(df_test_sample.user_id.values, df_test_sample.anime_id.values, df_test_sample.my_score.values.astype(float)))

            knn_baseline = KNNBaseline(sim_options = {'user_based' : False, 'name': 'pearson_baseline'})
            knn_baseline.fit(trainset)
            train_predictions = knn_baseline.test(trainset.build_testset())
            test_predictions = knn_baseline.test(testset)
            logging.info(f'Train RMSE :{accuracy.rmse(train_predictions)}')
            logging.info(f'Test RMSE :{accuracy.rmse(test_predictions)}')
            
            
            # storing predicted score of train and test dataset of KNNBaseline model
            train_knn_baseline_dict = dict()
            for user_id, anime_id, true_r, est, _ in train_predictions:
                if user_id in train_knn_baseline_dict:
                    train_knn_baseline_dict[user_id][anime_id] = est
                else:
                    train_knn_baseline_dict[user_id] = dict()
                    train_knn_baseline_dict[user_id][anime_id] = est
                    

            test_knn_baseline_dict = dict()
            for user_id, anime_id, true_r, est, _ in test_predictions:
                if user_id in test_knn_baseline_dict:
                    test_knn_baseline_dict[user_id][anime_id] = est
                else:
                    test_knn_baseline_dict[user_id] = dict()
                    test_knn_baseline_dict[user_id][anime_id] = est
            
            knn_baseline_rating_dict = {}

            # Add train predictions to the dictionary
            for user_id, anime_id, true_r, est, _ in train_predictions:
                key = (user_id, anime_id)
                if key not in knn_baseline_rating_dict:
                    knn_baseline_rating_dict[key] = []
                knn_baseline_rating_dict[key].append(est)

            # # Add test predictions to the dictionary
            # for user_id, anime_id, true_r, est, _ in test_predictions:
            #     key = (user_id, anime_id)
            #     if key not in knn_baseline_rating_dict:
            #         knn_baseline_rating_dict[key] = []
            #     knn_baseline_rating_dict[key].append(est)

            # # Average ratings for duplicates and finalize the dictionary
            # knn_baseline_rating_dict = {
            #     key: sum(ratings) / len(ratings) for key, ratings in knn_baseline_rating_dict.items()
            # }
            with open(self.hybrid_config.knnbaseline_pred_pth,'wb') as f:
                pickle.dump(knn_baseline_rating_dict,f)
            logging.info('collaborative completed!')
        except Exception as e:
            logging.info("error during training!")
            raise CustomException(e,sys)

        
    
    def predict(self,df,no_of_predictions=10):
        try:
            with open(self.content_based.content_config.type_vectorizer_pth,'rb') as f:
                type_vectorizer = pickle.load(f)
            with open(self.content_based.content_config.source_vectorizer_pth,'rb') as f:
                 source_vectorizer= pickle.load(f)
            with open(self.content_based.content_config.studio_vectorizer_pth,'rb') as f:
                 studio_vectorizer= pickle.load(f)
            with open(self.content_based.content_config.genre_vectorizer_pth,'rb') as f:
                genre_vectorizer = pickle.load(f)
            
            sample_encoded=sparse.load_npz(self.content_based.content_config.train_anime_encoded_pth)
            df_train=pd.read_csv(self.data_ing.data_ingestion_config.train_data_path)
            anime_profile=pd.read_csv(self.content_based.content_config.df_anime_profile_pth)
            sample_anime_id=anime_profile['anime_id'].values
            
            
            single_user_type_enc = type_vectorizer.transform(df['type'].values)
            single_user_source_enc = source_vectorizer.transform(df['source'].values)
            single_user_studio_enc = studio_vectorizer.transform(df['studio'].values)
            single_user_genre_enc = genre_vectorizer.transform(df['genre'].values)
            
            single_user_encoded = hstack((single_user_type_enc, single_user_source_enc, single_user_studio_enc, single_user_genre_enc)).tocsr()
            
            user_rating = df['my_score'].values
            user_vec = np.zeros(single_user_encoded.shape[1])
            for ind,vec in enumerate(single_user_encoded):
                # adding all the anime profile for a particular user by multiplying it with given user rating
                user_vec += vec.toarray()[0]*int(user_rating[ind]) 

            # computing cosine similarity between user profile and anime profile
            user_vec_normalize = normalize(user_vec.reshape(1,-1), norm = 'l2')
            similarity_vec = cosine_similarity(user_vec_normalize, sample_encoded)[0]
            scaler = MinMaxScaler(feature_range=(1, 10))
            content_based_user_ratings = scaler.fit_transform(similarity_vec.reshape(-1, 1)).ravel()
            
            test_anime=df['anime_id'].values
            train_user_id=df_train['user_id'].values
            with open(self.hybrid_config.knnbaseline_pred_pth,'rb') as f:
                knn_baseline_rating_dict = pickle.load(f)
            
            # computing hybrid recommender system predicted ratings
            user_id=df['user_id'].values[0]
            hybrid_user_ratings = []

            # Check if user is in the training data
            if user_id in train_user_id:
                for idx, anime_id in enumerate(test_anime):
                    # If anime ID exists in KNN Baseline predictions
                    if anime_id in knn_baseline_rating_dict.get(user_id, {}):
                        # Use a hybrid approach for existing anime
                        val = (
                            content_based_user_ratings[idx] * 0.03
                            + knn_baseline_rating_dict[user_id][anime_id]
                        )
                    else:
                        # Fallback to content-based ratings if not in KNN predictions
                        val = content_based_user_ratings[idx]
                    hybrid_user_ratings.append(val)
            else:  # For new custom users
                hybrid_user_ratings = content_based_user_ratings.tolist()
            
            hybrid_user_rating_sorted_index = np.array(hybrid_user_ratings).argsort()[::-1][1:]

            # Step 2: Get the list of anime IDs the user has already watched
            user_watched_anime_id = set(df['anime_id'].values)  # Convert to a set for faster lookup

            # Step 3: Initialize an empty list to store recommended anime IDs
            recommended_anime_id = []

            # Step 4: Iterate through sorted indices and filter out already-watched anime
            for i in hybrid_user_rating_sorted_index:
                # Check if the anime is not already watched
                if sample_anime_id[i] not in user_watched_anime_id:
                    recommended_anime_id.append(sample_anime_id[i])  # Add to recommendations

                # Stop when the desired number of predictions is reached
                if len(recommended_anime_id) == no_of_predictions:
                    break
            return recommended_anime_id
                
                     
                    
        except Exception as e:
            logging.info("error during prediction!")
            raise CustomException(e,sys)
        
        
        