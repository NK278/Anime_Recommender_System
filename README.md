

# Anime Recommendation System

An **Anime Recommendation System** built to recommend anime to users based on their preferences. This project employs **content-based** and **hybrid recommendation techniques** to deliver tailored recommendations. The app is developed with **Streamlit** for an interactive user interface.

## Project Structure

```
Anime-Recommendation-System/
│
├── app.py                     # Streamlit app to interact with the recommendation system
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup file
│
├── src/                       # Core source code
│   ├── components/            # Modular code for data ingestion, transformation, and recommendation logic
│   ├── constant/              # Definitions of reusable constants
│   ├── pipelines/             # Training pipeline implementation
│   ├── utils/                 # Utility functions and helper methods
│   └── exception.py, logger.py # Custom exception handling and logging
│
├── notebooks/                 # Jupyter notebooks for experimentation and model training
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── Model_Trainer.ipynb    # Model training and evaluation
│   └── merged_df.csv          # Dataset generated after preprocessing
│
├── artifacts/                 # Intermediate and final outputs (models, processed data)
│   ├── genre_vectorizer.pkl   # Genre vectorizer for content-based filtering
│   ├── knn_baseline_pred_dict.pkl # KNN model predictions
│   ├── user_profile.pkl       # User profiles
│   └── [Other artifacts]      # Pickled objects and data files for recommendation
│
├── logs/                      # Logs generated during execution
├── dist/                      # Distribution files for deployment
├── Dataset/                   # Original and preprocessed datasets
│   └── data/                  # CSV files of cleaned anime and user data
└── README.md                  # Project documentation
```

## Features

1. **Interactive Web App**: Built using Streamlit, allowing users to:
   - Explore anime datasets.
   - Get personalized recommendations.
2. **Hybrid Recommendation**: Combines content-based filtering with collaborative techniques for better accuracy.
3. **Data Pipeline**:
   - Preprocessing anime and user data (handled in `src/components/data_ingestion.py`).
   - Generating user and content profiles.
   - Training recommendation models.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/NK278/Anime-Recommendation-System.git](https://github.com/NK278/Anime_Recommender_System/tree/)
   cd Anime-Recommendation-System
   ```
2. Install dependencies:
   ```bash
   python setup.py install
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
## External Downloads

Since some files were too large to push to the repository, they are available for download via Google Drive:

- **Artifacts Folder**: [Download Artifacts](https://drive.google.com/drive/folders/1-H8yW2qPYr7XKmpGTi19JoBhwOmlRcYS?usp=drive_link)
- **Dataset Folder**: [Download Dataset](https://drive.google.com/drive/folders/1THfVGnuIOexG8xRRhxzupQqFP0Z1DRUv?usp=drive_link)
- **Merged Dataset (merged_df.csv)**: [Download merged_df.csv](https://drive.google.com/file/d/10jWy81NVpq00yKN8UJoHWxt54kdS50L-/view?usp=drive_link)


## How It Works

- **EDA and Model Training**:
  - Conducted in Jupyter notebooks (`notebooks/EDA.ipynb` and `notebooks/Model_Trainer.ipynb`).
  - Generates a processed dataset (`merged_df.csv`) for training.
  
- **Recommendation Engine**:
  - Uses `src/components/` for data ingestion and transformation.
  - Builds content profiles using vectorizers like `genre_vectorizer.pkl`.
  - Employs a hybrid approach combining content-based and collaborative filtering.

- **Streamlit App**:
  - Provides an interface for users to search and explore anime recommendations.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

