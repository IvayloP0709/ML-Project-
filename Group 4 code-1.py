#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# models
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Retrieve English stop words
stop_words = set(stopwords.words('english'))

# Define a text data preprocessing function
def preprocess_text(text):
    # Tokenize the text into a list of words
    words = word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word.lower() not in stop_words]
    # Concatenate the words into a string
    processed_text = ' '.join(words)
    return processed_text

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    
    #train data
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("empty") # using empty to replace missing value    
    # test data (for test dataset prediction)
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("empty")
    
    
    # transfer catergory data into numerical data
    logging.info("Feature engineering")
    train['year'] = pd.to_numeric(train['year'], errors='coerce', downcast='float') 
    
    # author ,editor contain list, inorder to concatanate, they are taken out from the list
    train['author'] = train['author'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    train['editor'] = train['editor'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # drop duplicates
    train = train.drop_duplicates(keep='first')
    
    # merge all text features together since they all text
    # train_features = train['ENTRYTYPE'] + ". " + train['title'] + ". " +train['publisher']+  ". " + train['abstract'] + ". " + train['editor']+ ". " + train['author']
    # Integrate all features into one
    train['all_features'] = train['ENTRYTYPE'] + ". " + train['title'] + ". " + train['publisher'] + ". "  + train['abstract'] + ". " + train['editor'] + ". " + train['author']
    
    train['all_features'] = train['all_features'].apply(preprocess_text)
    
    # split train dataset
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)

    
    # Apply TF-IDF transformation to all text features
    featurizer = ColumnTransformer(
        transformers=[("all_features", TfidfVectorizer(min_df = 0.0002
                                                       , max_df = 0.8
                                                       , ngram_range = (1,2)
                                                       , max_features=100000)
                                                       #, preprocessor=preprocess_text)
                       , "all_features")],
        remainder='drop')
        
    # apply models
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    ridge = make_pipeline(featurizer, Ridge())
    random_forest = make_pipeline(featurizer, RandomForestRegressor(n_estimators=100,  # Number of trees in the forest
                                                                    max_depth=None,    # Maximum depth of the tree (None means unlimited)
                                                                    min_samples_split=2,  # Minimum number of samples required to split an internal node
                                                                    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
                                                                    max_features='auto',  # Number of features to consider for the best split
                                                                    random_state=42))
    linear_regression = make_pipeline(featurizer, LinearRegression())
    lgbm = make_pipeline(featurizer, LGBMRegressor(boosting_type='gbdt',  # Gradient Boosting Decision Tree
                                                    num_leaves=31,         # Maximum tree leaves for base learners
                                                    learning_rate=0.05,    # Boosting learning rate
                                                    n_estimators=100,      # Number of boosting rounds
                                                    objective='regression',  # Regression task
                                                    random_state=42        # Random seed for reproducibility
                                                    )
                            )

    
    # Fit models
    logging.info("Fitting models")
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    ridge.fit(train.drop('year', axis=1), train['year'].values)
    random_forest.fit(train.drop('year', axis=1), train['year'].values)
    linear_regression.fit(train.drop('year', axis=1), train['year'].values)
    lgbm.fit(train.drop('year', axis=1), train['year'].values)
    
    # evaluate using MAE
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    err = mean_absolute_error(val['year'].values, ridge.predict(val.drop('year', axis=1)))
    logging.info(f"Ridge regress MAE: {err}")
    err = mean_absolute_error(val['year'].values, random_forest.predict(val.drop('year', axis=1)))
    logging.info(f"random_forest regress MAE: {err}")
    err = mean_absolute_error(val['year'].values, linear_regression.predict(val.drop('year', axis=1)))  # Evaluate Linear Regression
    logging.info(f"Linear regression MAE: {err}")
    err = mean_absolute_error(val['year'].values, lgbm.predict(val.drop('year', axis=1)))
    logging.info(f"LightGBM regress MAE: {err}")
    
    
    ####test dataset prediction
    logging.info(f"Predicting on test")
    
    #### Apply the same transformations as in the training data
    test['author'] = test['author'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    test['editor'] = test['editor'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
      
    # Create the 'all_features' column
    test['all_features'] = test['ENTRYTYPE'] + ". " + test['title'] + ". " + test['publisher'] + ". " + test['abstract'] + ". " + test['editor'] + ". " + test['author']
    
    test['all_features'] = test['all_features'].apply(preprocess_text)
    
    #### Predict using the best model
    pred = random_forest.predict(test)
    test['year'] = pred
    
    
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)
    
main()


# In[ ]:





# In[ ]:




