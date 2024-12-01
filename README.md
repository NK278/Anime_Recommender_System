# Anime Recommendation System  

![Anime Wallpaper](https://images.unsplash.com/photo-1625189659340-887baac3ea32?q=80&w=1373&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)  
*Unlock a world of anime recommendations tailored to your preferences!*  

An **Anime Recommendation System** designed to provide personalized anime recommendations based on user preferences. The project implements **content-based** and **hybrid recommendation techniques** to enhance accuracy and user satisfaction. Developed with **Streamlit**, the system offers an intuitive and interactive user interface for exploring and discovering anime.

---

## 📂 Project Structure  

```
Anime-Recommendation-System/  
│  
├── app.py                     # Streamlit app entry point  
├── requirements.txt           # Python dependencies  
├── setup.py                   # Project packaging and installation  
│  
├── src/                       # Core source code  
│   ├── components/            # Modular code for data ingestion, transformation, and recommendation logic  
│   ├── constant/              # Definitions of reusable constants  
│   ├── pipelines/             # Model training pipeline  
│   ├── utils/                 # Utility functions and helpers  
│   └── exception.py, logger.py # Custom exception handling and logging  
│  
├── notebooks/                 # Jupyter notebooks for experimentation  
│   ├── EDA.ipynb              # Exploratory Data Analysis  
│   ├── Model_Trainer.ipynb    # Model training and evaluation  
│   └── merged_df.csv          # Final preprocessed dataset  
│  
├── artifacts/                 # Intermediate and final outputs (models, processed data)  
│   ├── genre_vectorizer.pkl   # Genre vectorizer for content-based filtering  
│   ├── knn_baseline_pred_dict.pkl # KNN model predictions  
│   ├── user_profile.pkl       # User profiles for recommendations  
│   └── [Other artifacts]      # Additional serialized objects and data files  
│  
├── logs/                      # Execution logs  
├── dist/                      # Distribution files for deployment  
├── Dataset/                   # Original and preprocessed datasets  
│   └── data/                  # Raw and cleaned datasets  
└── README.md                  # Project documentation  
```

---

## 🎯 Features  

### 1. **Interactive Web Application**  
   - Built using **Streamlit**, allowing users to:  
     - Browse and explore the anime dataset.  
     - Receive personalized anime recommendations.  
   
### 2. **Hybrid Recommendation Engine**  
   - Combines **content-based filtering** (e.g., genre vectorization) with **collaborative techniques** (e.g., user-item interaction) for enhanced accuracy.  
   
### 3. **Comprehensive Data Pipeline**  
   - Automated ingestion and preprocessing of anime and user data (`src/components/data_ingestion.py`).  
   - Profile generation for users and content.  
   - Training and evaluation of recommendation models.  

---

## 🚀 Installation Guide  

### Step 1: Clone the Repository  
```bash  
git clone https://github.com/NK278/Anime-Recommendation-System.git  
cd Anime-Recommendation-System  
```  

### Step 2: Install Dependencies  
```bash  
python setup.py install  
```  

### Step 3: Run the Streamlit App  
```bash  
streamlit run app.py  
```  

---

## 📥 External Downloads  

Certain files are too large to store in the repository. They can be downloaded from the following links:  

1. **Artifacts Folder**: [Download Artifacts](https://drive.google.com/drive/folders/1-H8yW2qPYr7XKmpGTi19JoBhwOmlRcYS?usp=drive_link)  
2. **Dataset Folder**: [Download Dataset](https://drive.google.com/drive/folders/1THfVGnuIOexG8xRRhxzupQqFP0Z1DRUv?usp=drive_link)  
3. **Merged Dataset (`merged_df.csv`)**: [Download merged_df.csv](https://drive.google.com/file/d/10jWy81NVpq00yKN8UJoHWxt54kdS50L-/view?usp=drive_link)  

---

## ⚙️ How It Works  

### **1. Data Exploration and Model Training**  
- Conducted in Jupyter notebooks (`notebooks/EDA.ipynb` and `notebooks/Model_Trainer.ipynb`).  
- Produces a preprocessed dataset (`merged_df.csv`) for model training.  

### **2. Recommendation Engine**  
- Implements **content-based filtering** using vectorizers (e.g., `genre_vectorizer.pkl`).  
- Integrates **collaborative filtering** to enhance recommendations.  
- Encapsulates all logic within `src/components/`.  

### **3. Streamlit Interface**  
- Provides a user-friendly interface for:  
  - Searching anime by name or genre.  
  - Viewing tailored recommendations.  

---

## 🤝 Contributing  

We welcome contributions to enhance the Anime Recommendation System. Feel free to:  
- Submit issues for bugs or feature requests.  
- Open pull requests with improvements or fixes.  

---

## 📜 License  

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.  

---  
