# Predict Political Orientation from Twitter

The goal is to predict the political orientation from Twitter Contents. We defined Fratelli d'Italia, Forza Italia and Lega as right and Partito Democratico as left. 

Data Collection includes:
1. Tweets of the followers of the five major political parties
2. Tweets of random Twitter users

Phases:

# 1) Preparation of Dataset

Datasets
1. dataset with the political labeling agreement larger than 75%
2. dataset with 100 % of agreement

Selection on Twitter users that have at least one tweet
between June 1st and December 1st 2019

The users are annotated with different political labels:
1. the value 0 corresponding to left
2. the value 1 corresponding to right

# 2) Data Cleaning and Class Balancing

Steps of Data Cleaning:
1. Remove urls
2. Remove punctuation
3. Upcase/downcase
4. Remove emoji
5. Remove of Italian and English stop words

Class Balancing: 
a technique for dealing with highly unbalanced datasets is called undersampling,
which consists of removing samples from the majority class.

# 3) FastText for text representation and text classification

# 4) nlp models

word vectorizer tools: 

1.Bag-of-Words (BoW) 
2.Term Frequency–Inverse Document Frequency (TF-IDF)

Machine Learning classifiers:

1. Linear SVC (Support Vector Classifier)
2. Logistic Regression
3. Multinomial NB (Naive Bayes)
4. Random Forest Classifier
5. SGD Classifier (Stochastic Gradient Descent)
6. XGBoost Classifier




