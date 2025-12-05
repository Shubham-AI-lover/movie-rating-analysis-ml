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


