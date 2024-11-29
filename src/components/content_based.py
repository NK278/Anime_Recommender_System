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
from collections import defaultdict
import pickle
import tqdm
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

@dataclass
class Content_based_config:
     # categorical encoding
    type_vectorizer_pth=os.path.join('artifacts','type_vectorizer.pkl')
    source_vectorizer_pth=os.path.join('artifacts','source_vectorizer.pkl')
    studio_vectorizer_pth=os.path.join('artifacts','studio_vectorizer.pkl')
    genre_vectorizer_pth=os.path.join('artifacts','genre_vectorizer.pkl')
    
    # Anime profile
    train_anime_encoded_pth=os.path.join('artifacts','train_anime_encoded.npz')
    test_anime_encoded_pth=os.path.join('artifacts','test_anime_encoded.npz')
    user_train_anime_encoded_pth=os.path.join('artifacts','user_train_anime_encoded.npz')
    df_anime_profile_pth=os.path.join('artifacts','anime_profile.csv')
    
    # user_profile
    user_profile_pth=os.path.join('artifacts','user_profile.pkl')
    
    # user_watched_anime_dict
    user_watched_anime_dict_pth=os.path.join('artifacts','user_watched_anime_dict.pkl')
    
    # content based ratings dict
    content_based_ratings_dict_pth=os.path.join('artifacts','content_based_ratings_dict.pkl')

class Content_Based_Reccomendation:
    def __init__(self):
        self.utils=MainUtils()
        self.data_ing=DataIngestion()
        self.content_config=Content_based_config()
        self.data_trf=DataTransformation()
    
    def profile_create(self):
        try:
            df_train_sample=pd.read_csv(self.data_ing.data_ingestion_config.train_data_path)
            df_test_sample=pd.read_csv(self.data_ing.data_ingestion_config.test_data_path)
            
            df_train_anime_profile = df_train_sample.drop(['user_id','username','my_status','my_score','my_last_updated','gender'], axis = 1)
            df_train_anime_profile = df_train_anime_profile.drop_duplicates(subset = 'anime_id')
            
            # selecting categorical features from df_test_sample dataframe
            df_test_anime_profile = df_test_sample[['anime_id', 'title', 'type', 'source', 'studio', 'genre', 'episodes']]
            # Concatenating df_test_anime_profile with df_train_anime_profile
            df_test_anime_profile = pd.concat([df_test_anime_profile, df_train_anime_profile], ignore_index=True)
            # Dropping duplicates based on the 'anime_id' column
            df_test_anime_profile = df_test_anime_profile.drop_duplicates(subset='anime_id')
            
            df_test_anime_profile.to_csv(self.content_config.df_anime_profile_pth,index=False)
            
            
            type_vectorizer = CountVectorizer(lowercase=False)
            type_vectorizer.fit(df_train_anime_profile['type'].values)

            # Using the fitted CountVectorizer to convert the text to vectors
            train_type_enc = type_vectorizer.transform(df_train_anime_profile['type'].values)
            test_type_enc = type_vectorizer.transform(df_test_anime_profile['type'].values)
            user_train_type_enc = type_vectorizer.transform(df_train_sample['type'].values)
            
            logging.info("After vectorizations:")
            logging.info("Train type encoding shape: %s", train_type_enc.shape)
            logging.info("Test type encoding shape: %s", test_type_enc.shape)
            logging.info("User train type encoding shape: %s", user_train_type_enc.shape)
            logging.info("Feature names: %s", type_vectorizer.get_feature_names_out())
            
            
            # creating categorical encoding on 'source' feature
            source_vectorizer = CountVectorizer(lowercase = False, token_pattern=r'[\w\s\-]+')
            source_vectorizer.fit(df_train_anime_profile['source'].values)

            # we use the fitted CountVectorizer to convert the text to vector
            train_source_enc = source_vectorizer.transform(df_train_anime_profile['source'].values)
            test_source_enc = source_vectorizer.transform(df_test_anime_profile['source'].values)
            user_train_source_enc = source_vectorizer.transform(df_train_sample['source'].values)
            
            logging.info("After vectorizations:")
            logging.info("Train source encoding shape: %s", train_source_enc.shape)
            logging.info("Test source encoding shape: %s", test_source_enc.shape)
            logging.info("User train source encoding shape: %s", user_train_source_enc.shape)
            logging.info("Feature names: %s", source_vectorizer.get_feature_names_out())
            
            # creating categorical encoding on 'studio' feature
            studio_vectorizer = CountVectorizer(lowercase = False, token_pattern = r'[^,\s][^\,]*[^,\s]+')
            studio_vectorizer.fit(df_train_anime_profile['studio'].values)

            # we use the fitted CountVectorizer to convert the text to vector
            train_studio_enc = studio_vectorizer.transform(df_train_anime_profile['studio'].values)
            test_studio_enc = studio_vectorizer.transform(df_test_anime_profile['studio'].values)
            user_train_studio_enc = studio_vectorizer.transform(df_train_sample['studio'].values)
            
            logging.info("After vectorizations:")
            logging.info("Train studio encoding shape: %s", train_studio_enc.shape)
            logging.info("Test studio encoding shape: %s", test_studio_enc.shape)
            logging.info("User train studio encoding shape: %s", user_train_studio_enc.shape)
            logging.info("Feature names: %s", studio_vectorizer.get_feature_names_out())
            
            
            genre_vectorizer = CountVectorizer(lowercase = False, token_pattern = r'[^,\s][^\,]*[^,\s]*')
            genre_vectorizer.fit(df_train_anime_profile['genre'].values)

            # we use the fitted CountVectorizer to convert the text to vector
            train_genre_enc = genre_vectorizer.transform(df_train_anime_profile['genre'].values)
            test_genre_enc = genre_vectorizer.transform(df_test_anime_profile['genre'].values)
            user_train_genre_enc = genre_vectorizer.transform(df_train_sample['genre'].values)
            
            logging.info("After vectorizations:")
            logging.info("Train genre encoding shape: %s", train_genre_enc.shape)
            logging.info("Test genre encoding shape: %s", test_genre_enc.shape)
            logging.info("User train genre encoding shape: %s", user_train_genre_enc.shape)
            logging.info("Feature names: %s", genre_vectorizer.get_feature_names_out())
            
            with open(self.content_config.type_vectorizer_pth,'wb') as f:
                pickle.dump(type_vectorizer,f)
                
            with open(self.content_config.source_vectorizer_pth,'wb') as f:
                pickle.dump(source_vectorizer,f)
                
            with open(self.content_config.studio_vectorizer_pth,'wb') as f:
                pickle.dump(studio_vectorizer,f)
                
            with open(self.content_config.genre_vectorizer_pth,'wb') as f:
                pickle.dump(genre_vectorizer,f)
            
            train_anime_encoded = hstack((train_type_enc, train_source_enc, train_studio_enc, train_genre_enc)).tocsr()
            test_anime_encoded = hstack((test_type_enc, test_source_enc, test_studio_enc, test_genre_enc)).tocsr()
            user_train_anime_encoded = hstack((user_train_type_enc, user_train_source_enc, user_train_studio_enc, user_train_genre_enc)).tocsr()
            
            logging.info(f"Final Data matrix shape :{train_anime_encoded.shape}/n {test_anime_encoded.shape}/n {user_train_anime_encoded.shape}")

            sparse.save_npz(self.content_config.user_train_anime_encoded_pth,user_train_anime_encoded)
            sparse.save_npz(self.content_config.train_anime_encoded_pth,train_anime_encoded)
            sparse.save_npz(self.content_config.test_anime_encoded_pth,test_anime_encoded)
            
            logging.info("Starting of creating user profile")
            
            sample_user_list = np.unique(df_train_sample['user_id'].values)  # Finding all unique users in training data
            user_profile = dict()

            # Iterate over each unique user
            for user in sample_user_list:
                # Filter the dataframe for the current user
                user_df = df_train_sample[df_train_sample['user_id'] == user]
                user_rating = user_df['my_score'].values  # Ratings given by the user
                
                # Initialize the user profile vector
                user_vec = np.zeros(user_train_anime_encoded.shape[1])
                
                # Iterate over the rows of user_df to get encoded profiles and compute weighted sum
                for idx, row in user_df.iterrows():
                    # Find the index of the anime rated by the user in the original dataset
                    anime_index = row['anime_id']  # Assuming this is the identifier for anime
                    # Convert anime_index to the row number in the sparse matrix
                    anime_encoded_vec = user_train_anime_encoded[anime_index].toarray()[0]  # Access the encoded vector using index
                    # Multiply by the user's rating for that anime and add to user_vec
                    user_vec += anime_encoded_vec * int(row['my_score'])
                
                # Store the resulting user profile vector
                user_profile[user] = user_vec
            with open(self.content_config.user_profile_pth,'wb') as f:
                    pickle.dump(user_profile,f)
            logging.info(f"Number of user profiles:{len(user_profile)}")
            logging.info(f"Length of each user profile vector:{len(user_profile[list(user_profile.keys())[0]])}")
            logging.info(f"Type of user_profile: {type(user_profile)}")

            # # Iterate over the user_profile dictionary
            # for user, profile in user_profile.items():
            #     logging.info(f"User ID: {user} (Type: {type(user)})")
            #     logging.info(f"Profile Vector: {profile} (Type: {type(profile)})")
            #     if isinstance(profile, np.ndarray):
            #         logging.info(f"Profile Vector Data Type: {profile.dtype}")
            #         logging.info(f"Profile Vector Shape: {profile.shape}")
            #     else:
            #         logging.warning(f"Profile for user {user} is not a numpy.ndarray.")
            #     logging.info("------")
            
            
            logging.info("User profile completed")
            
                  # Initialize the dictionary to store user-anime mapping
            user_watched_anime_dict = {}
            sample_user_list = np.unique(
                np.concatenate([df_train_sample['user_id'].values, df_test_sample['user_id'].values])
            )

            # Iterate over each user
            for user in sample_user_list:
                # Get the list of anime rated by this user in the train dataset
                train_watched_anime = df_train_sample[df_train_sample['user_id'] == user]['anime_id'].tolist()
                
                # Get the list of anime rated by this user in the test dataset
                test_watched_anime = df_test_sample[df_test_sample['user_id'] == user]['anime_id'].tolist()
                
                # Combine the anime lists from train and test, ensuring uniqueness
                user_watched_anime_dict[user] = list(set(train_watched_anime + test_watched_anime))

            
            logging.info(f"Number of users in the dictionary: {len(user_watched_anime_dict)}")
            sample_user = list(user_watched_anime_dict.keys())[0]
            logging.info(f"Sample user: {sample_user}")
            logging.info(f"Anime watched by the sample user: {user_watched_anime_dict[sample_user]}")
            
            with open(self.content_config.user_watched_anime_dict_pth,'wb') as f:
                    pickle.dump(user_watched_anime_dict,f)
            
            
            logging.info("User and Anime profiles completed")
        except Exception as e:
            logging.info("Error during  building profiles training!")
            raise CustomException(e,sys)
        
    def content_train(self):
        try:
            df_train_sample=pd.read_csv(self.data_ing.data_ingestion_config.train_data_path)
            df_test_sample=pd.read_csv(self.data_ing.data_ingestion_config.test_data_path)
            df_train_anime_profile = df_train_sample.drop(['user_id','username','my_status','my_score','my_last_updated','gender'], axis = 1)
            df_train_anime_profile = df_train_anime_profile.drop_duplicates(subset = 'anime_id')
            
            # selecting categorical features from df_test_sample dataframe
            df_test_anime_profile = df_test_sample[['anime_id', 'title', 'type', 'source', 'studio', 'genre', 'episodes']]
            # Concatenating df_test_anime_profile with df_train_anime_profile
            df_test_anime_profile = pd.concat([df_test_anime_profile, df_train_anime_profile], ignore_index=True)
            # Dropping duplicates based on the 'anime_id' column
            df_test_anime_profile = df_test_anime_profile.drop_duplicates(subset='anime_id')
            
            with open(self.content_config.user_profile_pth,'rb') as f:
                user_profile=pickle.load(f)
            train_anime_encoded=sparse.load_npz(self.content_config.train_anime_encoded_pth)
            test_anime_encoded=sparse.load_npz(self.content_config.test_anime_encoded_pth)
            
            logging.info("Starting Content Based filtering computation.")
            
            sample_train_user_list = np.unique(df_train_sample['user_id'].values)
            train_anime_id_index = df_train_anime_profile['anime_id'].values 
            train_user_matrix = []

            for user in sample_train_user_list:
                user_profile_vec = user_profile[user] #getting the user profile vector for given user
                logging.info(f"Processing user: {user}")
                user_profile_normalize = normalize(user_profile_vec.reshape(1,-1), norm = 'l2') #normalizing the user profile vector
                logging.debug(f"Normalized profile vector for user {user}: {user_profile_normalize}")
        
                # computing cosine similarity between normalize user profile vector and anime profile matrix 
                similarity_vec = cosine_similarity(user_profile_normalize, train_anime_encoded)[0]
                logging.debug(f"Similarity vector for user {user}: {similarity_vec}") 
                scaler = MinMaxScaler(feature_range=(1, 10))
                train_user_matrix.append(scaler.fit_transform(similarity_vec.reshape(-1, 1)).ravel())
                logging.info(f"Completed similarity computation for user {user}.")

            train_user_matrix = np.array(train_user_matrix)

            train_content_based_df = pd.DataFrame(train_user_matrix, index = sample_train_user_list, columns = train_anime_id_index) 
            
            sample_train_user = df_train_sample['user_id'].values
            sample_train_anime = df_train_sample['anime_id'].values
            y_train_pred_content_based = []

            for i in range(len(sample_train_user)):
                y_train_pred_content_based.append(train_content_based_df[sample_train_anime[i]].loc[sample_train_user[i]])

            unique_test_user_list = np.unique(df_test_sample['user_id'].values)
            test_anime_id_index = df_test_anime_profile['anime_id'].values 


            # List to keep track of users for whom ratings are generated
            users_with_profiles = []

            # Iterate through each unique test user and compute their predicted ratings for each anime
            test_user_matrix = []
            for user in unique_test_user_list:
                if user in user_profile:  # Check if the user has a profile
                    user_profile_vec = user_profile[user]  # Get the user profile vector
                    user_profile_normalize = normalize(user_profile_vec.reshape(1, -1), norm='l2')  # Normalize the user profile vector

                    # Compute cosine similarity between normalized user profile vector and anime profile matrix 
                    similarity_vec = cosine_similarity(user_profile_normalize, test_anime_encoded)[0] 
                    scaler = MinMaxScaler(feature_range=(1, 10))
                    test_user_matrix.append(scaler.fit_transform(similarity_vec.reshape(-1, 1)).ravel())

                    # Track the user ID for which the profile was used
                    users_with_profiles.append(user)

            # Convert to numpy array
            test_user_matrix = np.array(test_user_matrix)

            # Create the dataframe containing the predicted ratings
            test_content_based_df = pd.DataFrame(test_user_matrix, index=users_with_profiles, columns=test_anime_id_index)
            
            new_sample_test_user = df_test_sample['user_id'].values
            new_sample_test_anime = df_test_sample['anime_id'].values

            y_test_pred_content_based = []
            for i in range(len(new_sample_test_user)):
                user = new_sample_test_user[i]
                anime = new_sample_test_anime[i]
                
                # Check if the user and anime exist in the predicted ratings DataFrame
                if user in test_content_based_df.index and anime in test_content_based_df.columns:
                    # If both exist, append the predicted rating
                    y_test_pred_content_based.append(test_content_based_df.at[user, anime])
                else:
                    # If either the user or anime is not found, use a default value (e.g., the mean rating)
                    y_test_pred_content_based.append(test_content_based_df.values.mean())

            # Convert to a numpy array if needed for further processing
            y_test_pred_content_based = np.array(y_test_pred_content_based)
            
            content_based_rating_dict = defaultdict(list)

            # Add train predictions to the dictionary
            for user, anime, rating in zip(sample_train_user, sample_train_anime, y_train_pred_content_based):
                content_based_rating_dict[(user, anime)].append(rating)

            # Add test predictions to the dictionary
            for user, anime, rating in zip(new_sample_test_user, new_sample_test_anime, y_test_pred_content_based):
                content_based_rating_dict[(user, anime)].append(rating)

            # Average ratings for duplicates and finalize the dictionary
            content_based_rating_dict = {
                key: sum(ratings) / len(ratings) for key, ratings in content_based_rating_dict.items()
            }

            with open(self.content_config.content_based_ratings_dict_pth, "wb") as f:
                pickle.dump(content_based_rating_dict,f)
            
            logging.info("Saving content Based ratings completed!")
            y_train=df_train_sample['my_score'].values
            y_test=df_test_sample['my_score'].values
            logging.info("In Content Based Filtering Model applied with Cosine Similarity : ")
            rmse_train, mape_train = self.utils.get_error_metrics(y_train.astype(float), y_train_pred_content_based)
            logging.info(f'Train RMSE :  {rmse_train}')

            rmse_test, mape_test = self.utils.get_error_metrics(y_test.astype(float), y_test_pred_content_based)
            logging.info(f'Test RMSE :  {rmse_test}')
            logging.info(f'Test MAPE :  {mape_test}')
            
            
            
        except Exception as e:
            logging.info("Error during content based rec training!")
            raise CustomException(e,sys)
    
    
    