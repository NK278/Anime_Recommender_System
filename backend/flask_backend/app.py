from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load data and models
with open('../../notebooks/anime_id_and_title_dict.pkl', 'rb') as f:
    anime_id_and_title_dict = pickle.load(f)

with open('../../notebooks/user_profile.pkl', 'rb') as f:
    user_profile = pickle.load(f)

with open('../../notebooks/anime_profiles.pkl', 'rb') as f:
    anime_profiles = pickle.load(f)

with open('../../notebooks/similar_anime_dict.pkl', 'rb') as f:
    similar_anime_dict = pickle.load(f)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json.get('userInput')
    user_id = int(user_input)  # Assuming user_input is the user ID

    if user_id not in user_profile:
        return jsonify({'error': 'User not found'}), 404

    user_vec = user_profile[user_id]
    similarity_scores = cosine_similarity([user_vec], anime_profiles)[0]
    top_anime_indices = similarity_scores.argsort()[::-1][:10]
    recommendations = [{'title': anime_id_and_title_dict[idx]} for idx in top_anime_indices]

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)