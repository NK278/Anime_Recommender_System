import os, sys
from src.components.data_ingestion import DataIngestion
from src.components.content_based import Content_Based_Reccomendation
from src.components.hybrid_rec import Hybrid_Reccomendation
from src.logger import logging
from src.exception import CustomException

if __name__=="__main__":
    # # Data Ingestion
    data_ing_obj=DataIngestion()
    train_path,test_path=data_ing_obj.initiate_data_ingestion()
    print(train_path,' ',test_path)
    
    # Content Based Reccomendations
    content_based_obj=Content_Based_Reccomendation()
    content_based_obj.profile_create()
    content_based_obj.content_train()
    
    # Collaborative Filtering
    collaborative_filtering_obj=Hybrid_Reccomendation()
    collaborative_filtering_obj.train_collaborative()
    