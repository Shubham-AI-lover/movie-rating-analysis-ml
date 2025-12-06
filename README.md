# 🎬 Movie Ratings Analysis & Prediction Using Python  
## A complete EDA + Regression-based prediction project  
## 1. Introduction  

This project is a beginner-friendly Machine Learning model that predicts movie ratings based on a given userId and movieId. 
The goal is to understand how recommendation systems work at a basic level using Python, Pandas, and Scikit-Learn.  

I used a cleaned and processed dataset **sample_df (derived from movie_df)** and built a regression model to estimate rating values.   

The goal is to analyze movie rating behavior  
Understand trends in genres, ratings, and users  
Build a baseline machine learning model to predict ratings  
Dataset used: MovieLens (movies.csv + ratings.csv)  
Final dataset after merging + cleaning: movies_df  
ML used on smaller sampled dataset: sample_df (~400k rows)  

<img width="781" height="756" alt="Movie Rating Data Analysis - visual selection" src="https://github.com/user-attachments/assets/339b6186-693c-4c54-92fe-a4c06e8efc6c" />


## 2. Dataset Description

The dataset contains the following key columns:

userId – Unique ID for each user  
movieId – Unique ID for each movie  
rating – Rating given by the user (0.5 to 5.0)  
I sampled the dataset (sample_df) to make analysis and modelling faster.    

**movies.csv:** movieId, title, genres  
**ratings.csv:** userId, movieId, rating, timestamp  

I performed:  

Merging datasets  
Extracting year using regex  
Splitting genres  
Converting timestamp to date  

## 3 Data Cleaning  

Extracted **year** from title using regex  
Handled **405 missing year** values using median   
Converted timestamp → datetime  
Split genres column into lists  
Renamed columns  
Created a clean dataset **(movies_df)**  

## 4 EDA (Exploratory Analysis)  

Rating distribution  
Count of ratings over time  
Genre frequency  
Average rating by genre  
Trends in movie release years  

### (A)  
<img width="925" height="588" alt="3" src="https://github.com/user-attachments/assets/fce1a92d-b898-42ad-bf29-d5ae209ab08e" />  

### Key insights  
Bars around 4 and 5 are **tallest**, so most users gave positive ratings.  
Ratings near 1 and 2 have very short bars, meaning very few users rated that low.    
The smooth blue curve (KDE line) overlays the histogram to show the overall distribution shape, which peaks near **rating 4**, confirming that user satisfaction is generally **high**. 

### (B)  
<img width="1154" height="662" alt="4" src="https://github.com/user-attachments/assets/04d27914-2f21-4d85-b9c1-8729b35ab9e9" />  

Early years (pre‑1920) are quite volatile, probably because there are fewer movies, so each one affects the average a lot.    
From roughly the 1920s to the 1960s, the average rating stays fairly high and stable, mostly in the upper 3s to around 4.  

After about the late 1970s or 1980s, the line gradually drifts downward, suggesting that more recent movies in this dataset receive slightly lower average ratings.  
The sharp drop at the very end (around the most recent year) is likely due to very few movies in that last year  

### (c)  
<img width="1337" height="714" alt="5" src="https://github.com/user-attachments/assets/8cb89503-e4ab-4cb0-99ba-7526cdaf3d20" />  

Early years have very few ratings, indicating low platform usage or fewer recorded movies.  
The number of ratings rises sharply, peaks around the late 1990s to early 2000s, then gradually declines, suggesting a boom period when users were most active followed by reduced activity in later years.  

### (D)  
<img width="1439" height="704" alt="7" src="https://github.com/user-attachments/assets/78e29c5f-6636-4220-873b-6421643b4c2e" />  

Most genres fluctuate around ratings of roughly 3 to 4, meaning that, on average, viewers rate movies similarly across genres, without any single genre consistently dominating.  
Some years have sharp spikes or drops for particular genres, which often happen when there are only a few movies of that genre in those years, making the average unstable.  

Overall, there is no strong upward or downward trend for any one genre; ratings remain relatively stable over time, suggesting that genre popularity in terms of rating is fairly consistent across decades.  

### (E) 
<img width="1023" height="608" alt="11" src="https://github.com/user-attachments/assets/7b0a5887-72cf-4206-b21d-c7f3a3097e48" /><img width="1118" height="921" alt="12" src="https://github.com/user-attachments/assets/0e03a319-edf2-4077-a7f8-6ebfc590eef0" />  

Drama and Comedy dominate the dataset, suggesting that these genres are produced significantly more often. Action and Thriller also show strong representation. On the other hand, niche genres like Children and Fantasy are far less common, indicating smaller production volume.  

While Film-Noir and Documentary genres are produced far less frequently, they receive the highest average ratings, suggesting that niche genres tend to be more critically appreciated. Drama appears in both charts, indicating it is not only widely produced but also relatively well-rated.  

There is a clear difference between **popularity (frequency) and quality (ratings).**  
  
**Popular genres**: Drama, Comedy, Action, Thriller  
**Highly rated genres**: Film-Noir, War, Documentary, Crime  

This means:  
  
Audiences appreciate certain niche genres more.  
Highly produced genres do not always receive the highest ratings.  
Drama is the only genre that is both high in count and above-average in rating — showing its strong overall performance.  

## 5 Feature Engineering  

**Added critical predictive features**:  
   movie_avg_rating  
   user_avg_rating  
   movie_rating_count  
   user_rating_count  

**Extracted**:  
   Movie year  
   Rating year  
   Rating month  
   
**Purpose**: Improve model accuracy  

 ## 6 Machine Learning Approach  

**Objective**: Predict movie ratings using supervised learning  

**Models used**:
  Linear Regression  
  Ridge Regression  
  Lasso Regression    
  
**Split**: 80% train / 20% test  

**Encoding**: OrdinalEncoder with unknown_value=-1  

## 7 Model Development Steps  

Training features selected  
Handled unseen user/movie IDs  
Trained Linear, Ridge, and Lasso models  
Evaluated using Root Mean Squared Error (RMSE)  
Compared three models side-by-side  

## 8 Model Evaluation (RMSE Results)
**Model           	RMSE**  
Linear Regression	0.7546  
Ridge Regression	0.7546  
Lasso Regression	0.7547  

**Interpretation:**  
Regression models perform similarly due to clean numeric feature space  
RMSE ≈ 0.75 → Very strong baseline for rating prediction  
Indicates consistent model performance  
Dataset is simple → regularization makes little difference  
Good baseline before using advanced recommenders (SVD, NCF, LightFM)  

## Key Insights from ML  

 Users and movies have strong rating patterns  
 Movie and user averages are powerful predictors  
 Linear models work very well for a basic recommender  
 Ridge/Lasso do not improve much due to simple feature space  

 ## 📦 Technologies Used  

Python  
Pandas, NumPy  
Matplotlib, Seaborn  
Scikit-learn  
Jupyter Notebook  

## 🏁 Next Steps (Future Improvements)

You can extend this project by adding:  

🔸 **Collaborative Filtering (SVD)**  

For better predictions.  

🔸 **Deep Learning Models**  

Neural Collaborative Filtering (NCF).  

🔸 **Web App Deployment**  

Using Streamlit or Flask.  

🔸 **More Feature Engineering**  

Incorporate genre encoding, sentiment from titles, movie age, etc.  



### Author  
Shubham Sharma  
Data Analyst | Python Enthusiast | ML Beginner  

<img width="360" height="217" alt="Thank you" src="https://github.com/user-attachments/assets/adf398f6-d14a-400a-bc3a-5f07287d2fa3" />




