import streamlit as st
import pickle
import os
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from src.components.hybrid_rec import Hybrid_Reccomendation

# Set paths to pre-stored dictionary files
artifacts_dir = "artifacts"
anime_images_dict_path = os.path.join(artifacts_dir, "anime_images_dict.pkl")
anime_title_dict_path = os.path.join(artifacts_dir, "anime_title_dict.pkl")
user_rated_anime_dict_path = os.path.join(artifacts_dir, "user_rated_anime_dict.pkl")
train_user_id_path = os.path.join(artifacts_dir, "train_user_id.pkl")
anime_title_list_path = os.path.join(artifacts_dir, "anime_title_list.pkl")
user_watched_anime_dict_path = os.path.join(artifacts_dir, "user_watched_anime_dict.pkl")
df_sample_anime_profile_path = os.path.join(artifacts_dir, "anime_profile.csv")

# Initialize Hybrid Recommendation System
hybrid_rec = Hybrid_Reccomendation()

# Load pre-stored dictionaries
with open(anime_images_dict_path, "rb") as f:
    anime_images_dict = pickle.load(f)

with open(anime_title_dict_path, "rb") as f:
    anime_title_dict = pickle.load(f)

with open(user_rated_anime_dict_path, "rb") as f:
    user_rated_anime_dict = pickle.load(f)

with open(train_user_id_path, "rb") as f:
    train_user_id = pickle.load(f)

with open(anime_title_list_path, "rb") as f:
    anime_title_list = pickle.load(f)

with open(user_watched_anime_dict_path, "rb") as f:
    user_watched_anime_dict = pickle.load(f)

# Load the user-anime profile DataFrame
df_sample_anime_profile = pd.read_csv(df_sample_anime_profile_path)

# Define the scraper function
def get_anime_image_urls(anime_ids):
    anime_images = {}
    for anime_id in anime_ids:
        try:
            url = f"https://myanimelist.net/anime/{anime_id}"
            response = requests.get(url)
            if response.status_code != 200:
                anime_images[anime_id] = None
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            image_tag = soup.find('img', {'itemprop': 'image'})
            if image_tag:
                anime_images[anime_id] = image_tag['data-src']
            else:
                anime_images[anime_id] = None
        except Exception as e:
            anime_images[anime_id] = None
    return anime_images

# Streamlit UI
st.title("Hybrid Anime Recommender System")

# **Section 1: User Recommendation**
st.header("Anime Recommendations for Existing Users")
user_id = st.text_input("Enter User ID", "")
max_watched_display = 6  # Limit the number of watched anime displayed to 6

if user_id:
    try:
        user_id = int(user_id)
        if user_id not in user_watched_anime_dict:
            st.error(f"User ID {user_id} not found!")
        else:
            watched_anime_list = user_watched_anime_dict[user_id]
            df_user = pd.concat(
                [df_sample_anime_profile[df_sample_anime_profile['anime_id'] == anime_id] for anime_id in watched_anime_list],
                ignore_index=True,
            )
            df_user['my_score'] = df_user['anime_id'].map(user_rated_anime_dict).fillna(0).astype(int)
            df_user['user_id'] = user_id
            recommended_anime_ids = hybrid_rec.predict(df_user)

            # Fetch missing images for recommended anime
            missing_image_ids = [anime_id for anime_id in recommended_anime_ids if anime_images_dict.get(anime_id) is None]
            if missing_image_ids:
                st.info(f"Fetching images for {len(missing_image_ids)} recommended anime...")
                scraped_images = get_anime_image_urls(missing_image_ids)
                anime_images_dict.update(scraped_images)

            recommended_anime_dict = {
                int(anime_id): {
                    "title": anime_title_dict.get(int(anime_id), "Unknown Title"),
                    "image_url": anime_images_dict.get(int(anime_id), None),
                }
                for anime_id in recommended_anime_ids
            }

            # Process watched anime
            df_user = df_user.sort_values(by="my_score", ascending=False).head(max_watched_display)
            watched_anime_list = df_user["anime_id"].tolist()

            watched_missing_image_ids = [anime_id for anime_id in watched_anime_list if anime_images_dict.get(anime_id) is None]
            if watched_missing_image_ids:
                st.info(f"Fetching images for {len(watched_missing_image_ids)} watched anime...")
                scraped_watched_images = get_anime_image_urls(watched_missing_image_ids)
                anime_images_dict.update(scraped_watched_images)

            watched_anime_dict = {
                int(anime_id): {
                    "title": anime_title_dict.get(int(anime_id), "Unknown Title"),
                    "image_url": anime_images_dict.get(int(anime_id), None),
                    "rating": int(df_user.loc[df_user["anime_id"] == anime_id, "my_score"].values[0]),
                }
                for anime_id in watched_anime_list
            }

            # Display recommended anime in rows
            st.subheader("Recommended Anime for You")
            recommended_cols = st.columns(3)  # Display 3 images per row
            for i, anime in enumerate(recommended_anime_dict.values()):
                with recommended_cols[i % 3]:
                    if anime["image_url"]:
                        st.image(anime["image_url"], width=150, caption=anime["title"])
                    else:
                        st.write(f"**{anime['title']}** (Image not available)")

            # Display watched anime in rows
            st.subheader(f"Your Watched Anime (Showing up to {max_watched_display})")
            watched_cols = st.columns(3)  # Display 3 images per row
            for i, anime in enumerate(watched_anime_dict.values()):
                with watched_cols[i % 3]:
                    if anime["image_url"]:
                        st.image(
                            anime["image_url"],
                            width=150,
                            caption=f"{anime['title']} (Rating: {anime['rating']})",
                        )
                    else:
                        st.write(f"**{anime['title']}** (Rating: {anime['rating']}) - Image not available")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# **Section 2: Custom User**
st.header("Create Custom User")
custom_user_data = st.text_area(
    "Enter custom user data in the format: `Anime_1: 10 | Anime_2: 9`",
    placeholder="e.g., Anime_1: 10 | Anime_2: 9",
)

if st.button("Submit Custom User Data"):
    try:
        # Parse the custom user input
        split_custom_user_ratings = custom_user_data.split("|")
        anime_rating_dict = {}
        for entry in split_custom_user_ratings:
            title_and_ratings = entry.split(":")
            anime_rating = int(title_and_ratings[-1].strip())
            anime_title = ":".join(title_and_ratings[:-1]).strip()
            anime_rating_dict[anime_title] = anime_rating

        # Merge with the anime profile data
        custom_df = pd.DataFrame(anime_rating_dict.items(), columns=["title", "my_score"])
        custom_df = pd.merge(custom_df, df_sample_anime_profile, on="title", how="inner")

        if custom_df.empty:
            st.error("None of the entered anime titles match the dataset. Please check your input.")
        else:
            # Assign a dummy user ID
            custom_df["user_id"] = [1] * len(custom_df)

            # Generate recommendations using the predict method
            recommended_anime_ids = hybrid_rec.predict(custom_df)

            # Fetch missing images for recommended anime
            recommended_missing_image_ids = [anime_id for anime_id in recommended_anime_ids if anime_images_dict.get(anime_id) is None]
            if recommended_missing_image_ids:
                st.info(f"Fetching images for {len(recommended_missing_image_ids)} recommended anime...")
                scraped_recommended_images = get_anime_image_urls(recommended_missing_image_ids)
                anime_images_dict.update(scraped_recommended_images)

            recommended_anime_dict = {
                int(anime_id): {
                    "title": anime_title_dict.get(int(anime_id), "Unknown Title"),
                    "image_url": anime_images_dict.get(int(anime_id), None),
                }
                for anime_id in recommended_anime_ids
            }

            # Sort watched anime and limit to the top 5
            custom_df = custom_df.sort_values(by=["my_score"], ascending=False).head(5)

            # Fetch missing images for watched anime
            watched_anime_list = custom_df["anime_id"].tolist()
            watched_missing_image_ids = [anime_id for anime_id in watched_anime_list if anime_images_dict.get(anime_id) is None]
            if watched_missing_image_ids:
                st.info(f"Fetching images for {len(watched_missing_image_ids)} watched anime...")
                scraped_watched_images = get_anime_image_urls(watched_missing_image_ids)
                anime_images_dict.update(scraped_watched_images)

            watched_anime_dict = {
                int(anime_id): {
                    "title": anime_title_dict.get(int(anime_id), "Unknown Title"),
                    "image_url": anime_images_dict.get(int(anime_id), None),
                    "rating": int(custom_df.loc[custom_df["anime_id"] == anime_id, "my_score"].values[0]),
                }
                for anime_id in watched_anime_list
            }

            # Display recommended anime in rows
            st.subheader("Recommended Anime for Custom User")
            recommended_cols = st.columns(3)  # Display 3 images per row
            for i, anime in enumerate(recommended_anime_dict.values()):
                with recommended_cols[i % 3]:
                    if anime["image_url"]:
                        st.image(anime["image_url"], width=150, caption=anime["title"])
                    else:
                        st.write(f"**{anime['title']}** (Image not available)")

            # Display watched anime in rows
            st.subheader("Watched Anime for Custom User (Top 5)")
            watched_cols = st.columns(3)  # Display 3 images per row
            for i, anime in enumerate(watched_anime_dict.values()):
                with watched_cols[i % 3]:
                    if anime["image_url"]:
                        st.image(
                            anime["image_url"],
                            width=150,
                            caption=f"{anime['title']} (Rating: {anime['rating']})",
                        )
                    else:
                        st.write(f"**{anime['title']}** (Rating: {anime['rating']}) - Image not available")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# **Section 3: Anime Title Autocomplete**
st.header("Search Anime Title")
anime_search = st.text_input("Search for an Anime Title", "")
if anime_search:
    filtered_titles = [title for title in anime_title_list if anime_search.lower() in title.lower()]
    st.write("Matching Titles:")
    st.write(filtered_titles[:10])

# **Optional: Show Full Anime Titles**
if st.checkbox("Show All Anime Titles"):
    st.write(anime_title_list)
