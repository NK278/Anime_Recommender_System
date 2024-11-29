import pickle
import pandas as pd
import os
import numpy as np

# Initialize data paths (replace with actual paths)
train_data_path = os.path.join('artifacts','train.csv')
# anime_images_path = os.path.join('artifacts','anime_images.csv')
# raw_data_path = os.path.join('artifacts','raw.csv')
# df_sample_anime_profile_path =os.path.join('artifacts','anime_profile.csv')
# Load data
df_train = pd.read_csv(train_data_path)
# anime_images = pd.read_csv(anime_images_path)
# raw = pd.read_csv(raw_data_path)

# Create dictionaries
# anime_images_dict = anime_images.set_index('anime_id')['image_url'].to_dict()
# anime_title_dict = df_train.set_index('anime_id')['title'].to_dict()
# user_rated_anime_dict = df_train.set_index('user_id')['my_score'].to_dict()
# train_user_id = df_train['user_id'].astype(str).tolist()
# anime_title_list = df_train['title'].dropna().unique().tolist()

# # Save dictionaries to files
# with open(os.path.join('artifacts','anime_images_dict.pkl'), "wb") as f:
#     pickle.dump(anime_images_dict, f)

# with open(os.path.join('artifacts','anime_title_dict.pkl'), "wb") as f:
#     pickle.dump(anime_title_dict, f)

# with open(os.path.join('artifacts','user_rated_anime_dict.pkl'), "wb") as f:
#     pickle.dump(user_rated_anime_dict, f)

# with open(os.path.join('artifacts','train_user_id.pkl'), "wb") as f:
#     pickle.dump(train_user_id, f)

# with open(os.path.join('artifacts','anime_title_list.pkl'), "wb") as f:
#     pickle.dump(anime_title_list, f)
import pickle
import os
from bs4 import BeautifulSoup
import requests

# Load existing anime_images_dict
artifacts_dir = "artifacts"
anime_images_dict_path = os.path.join(artifacts_dir, "anime_images_dict.pkl")

with open(anime_images_dict_path, "rb") as f:
    anime_images_dict = pickle.load(f)

# Define the scraper function
def get_anime_image_urls(anime_ids):
    anime_images = {}
    for anime_id in anime_ids:
        try:
            url = f"https://myanimelist.net/anime/{anime_id}"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch data for anime_id {anime_id}")
                anime_images[anime_id] = None
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            image_tag = soup.find('img', {'itemprop': 'image'})
            if image_tag:
                anime_images[anime_id] = image_tag['data-src']
            else:
                print(f"No image found for anime_id {anime_id}")
                anime_images[anime_id] = None
        except Exception as e:
            print(f"Error processing anime_id {anime_id}: {e}")
            anime_images[anime_id] = None
    return anime_images

# Find missing image URLs
missing_image_ids = [anime_id for anime_id, url in anime_images_dict.items() if not url]

# Scrape images for missing IDs
scraped_images = get_anime_image_urls(missing_image_ids)

# Update the existing anime_images_dict with scraped URLs
anime_images_dict.update(scraped_images)

# Save the updated anime_images_dict
with open(anime_images_dict_path, "wb") as f:
    pickle.dump(anime_images_dict, f)

print("Updated anime_images_dict with scraped URLs.")
