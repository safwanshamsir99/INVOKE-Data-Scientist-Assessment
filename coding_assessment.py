# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:56:39 2023

@author: Safwan Shamsir
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

#%% FUNCTION
def change_values(val):
    '''
    This function is to categorize B for income below 5K, 
    M for income 5K and above.

    Parameters
    ----------
    val : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if val in ['Less than 1K', '1K to 2K', '2K to 3K', '3K to 4K', '4K to 5K']:
        return 'B'
    elif val in ['5K to 6K', '6K to 7K', '7K to 8K', '8K to 9K', '9K to 10K', '10K or more']:
        return 'M'
    else:
        return val

def plot_cat(df,categorical_col):
    '''
    This function is to generate plots for categorical columns

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    categorical_col : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for i in categorical_col:
        plt.figure() 
        sns.countplot(df[i]) 
        plt.show()

def plot_con(df,continuous_col):
    '''
    This function is to generate plots for continuous columns

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    continuous_col : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for j in continuous_col:
        plt.figure()
        sns.distplot(df[j])
        plt.show()

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,         
        Journal of the Korean Statistical Society 42 (2013): 323-328    
    """    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n    
    r,k = confusion_matrix.shape    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% STATIC
CSV_PATH = os.path.join(os.getcwd(),'surveyA.csv')
BEST_PIPE_PATH = os.path.join(os.getcwd(),'fine_tune.pkl')
MODEL_PATH = os.path.join(os.getcwd(),'best_model_ft.pkl')

#%% DATA LOADING
df = pd.read_csv(CSV_PATH)

#%% DATA INSPECTION
df.info() # A lot of null values
df.duplicated().sum() # 6 duplicated data
df.isna().sum() # a lot of null values (need to be imputed since this is a small dataset)
stats = df.describe().T
df.boxplot() #education loan has an outlier but it will not be removed yet

df_backup = df
df_backup.drop('education_loan', axis=1, inplace=True)
df_backup.boxplot() 
'''
Other continuous columns also have outlier but will not be removed since
the outliers may contain informations.
'''

df['salary'].unique()
column_names = df.columns

# Change the category in salary column into B (<5k) or M(>5k)
df['salary'] = df['salary'].apply(change_values)

cat_data = df.columns[df.dtypes=='object']
for cat in cat_data:
    print(df[cat].unique())

#%% Data visualization
con_data = df.columns[(df.dtypes=='int64') | (df.dtypes=='float64')]
plot_con(df, con_data) # graphs are skewed

cat_data = df.columns[df.dtypes=='object']
plot_cat(df, cat_data) 
'''
Imbalance categorical data especially with the target column (salary)
where the M category only has about 250 rows while B category has more
1750 rows.
'''
#%% DATA CLEANING
# drop duplicated rows
df = df.drop_duplicates()
df.duplicated().sum() # Checked = 0 duplicate data
'''
Dropping the house_value column since 43% (957 out of 2226 rows) of the 
column are null values considering this is a small dataset, 
hence it will make the ml model training inaccurate.
'''

df = df.drop(labels='house_value',axis=1)
column_names = df.columns
cat_data = df.columns[df.dtypes=='object']

# Impute the NaN in categorical column by using mode
for col in df.columns:
    if df[col].dtypes == 'object':  # Encode only object columns
        df[col] = df[col].fillna(df[col].mode()[0])

#%% Label encoding the categorical column to do MICE method
for col in df.columns:
    if df[col].dtypes == 'object':  # Encode only object columns
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_path = os.path.join(os.getcwd(),f'le_{col}.pkl')
        with open(le_path,'wb') as file:
            pickle.dump(le,file)

# MICE imputation for continuous column because graphs are skewed
ii_imputer = IterativeImputer()
imputed_data_ii = ii_imputer.fit_transform(df)
df = pd.DataFrame(imputed_data_ii)
df.columns = column_names

#%% FEATURES SELECTION
# categorical features vs categorical target using cramer's V
for cat in cat_data:
    print(cat)
    confussion_mat = pd.crosstab(df[cat],df['salary']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))
    
'''
Based on the cramer's V stats, no categorical columns that will be selected as 
for model training since they have a low correlation to the target (<0.5).
Nevertheless, chi-square test can be done to look for the relationship between
the features to the target.
'''

# continuous features vs categorical target using LogisticRegression
for con in con_data:
    logreg = LogisticRegression()
    logreg.fit(np.expand_dims(df[con],axis=-1),df['salary'])
    print(con)
    print(logreg.score(np.expand_dims(df[con],axis=-1),df['salary']))
'''
All of the continuous columns have a high correlation (>0.8) with the target.
Therefore, all of the continuous column will be selected as features.
'''

#%% Preprocessing
X = df.loc[:,con_data]
y = df.loc[:,'salary']

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.2,
                                                 random_state=3)
'''
80% of the dataset will be used for model training, and 20% will be used 
for model testing.
'''

#%% Machine learning development
'''
To automate the process of choosing the best machine learning model for 
this dataset, machine learning pipeline will be used. The machine learning 
models that will be tested are Logistic Regression, Random Forest Classifier,
Decision Tree Classifier, K-Neighbors Classifier, and Support Vector 
Classifier.
'''
pl_std_lr = Pipeline([('Standard Scaler',StandardScaler()),
                      ('LogClassifier',LogisticRegression())]) 

pl_mm_lr = Pipeline([('Min Max Scaler',MinMaxScaler()),
                     ('LogClassifier',LogisticRegression())])

#RF
pl_std_rf = Pipeline([('Standard Scaler',StandardScaler()),
                      ('RFClassifier',RandomForestClassifier())]) 

pl_mm_rf = Pipeline([('Min Max Scaler',MinMaxScaler()),
                     ('RFClassifier',RandomForestClassifier())]) 

# Decision Tree
pl_std_tree = Pipeline([('Standard Scaler',StandardScaler()),
                        ('DTClassifier',DecisionTreeClassifier())]) 

pl_mm_tree = Pipeline([('Min Max Scaler',MinMaxScaler()),
                       ('DTClassifier',DecisionTreeClassifier())]) 

# KNeighbors
pl_std_knn = Pipeline([('Standard Scaler',StandardScaler()),
                       ('KNClassifier',KNeighborsClassifier())]) 

pl_mm_knn = Pipeline([('Min Max Scaler',MinMaxScaler()),
                      ('KNClassifier',KNeighborsClassifier())])

# SVC
pl_std_svc = Pipeline([('Standard Scaler',StandardScaler()),
                       ('SVClassifier',SVC())]) 

pl_mm_svc = Pipeline([('Min Max Scaler',MinMaxScaler()),
                      ('SVClassifier',SVC())])

# create pipeline
pipelines = [pl_std_lr,pl_mm_lr,pl_std_rf,pl_mm_rf,pl_std_tree,
             pl_mm_tree,pl_std_knn,pl_mm_knn,pl_std_svc,pl_mm_svc]

# fitting the data
for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_dict = {0:'SS+LR', 
             1:'MM+LR',
             2:'SS+RF',
             3:'MM+RF',
             4:'SS+Tree',
             5:'MM+Tree',
             6:'SS+KNN',
             7:'MM+KNN',
             8:'SS+SVC',
             9:'MM+SVC'}
best_accuracy = 0

# model evaluation
scores = []
model_names = list(pipe_dict.values())
for i,model in enumerate(pipelines):
  scores = [model.score(X_test, y_test) for model in pipelines]
  if model.score(X_test, y_test) > best_accuracy:
    best_accuracy = model.score(X_test,y_test)
    best_pipeline = model
    best_scaler = pipe_dict[i]
  result = pd.DataFrame({
      'Model':model_names,
      'Score':scores})
print(result)
print('The best ml pipeline for this dataset will be {} with accuracy of {}'
      .format(best_scaler, best_accuracy))

#%% Fine tuning process
'''
Based on the pipeline, model with highest accuracy is Standard Scaler 
and Support Vector Classifier with accuracy of 0.914.
'''
pl_std_svc = Pipeline([('Standard Scaler',StandardScaler()),
                       ('SVClassifier',SVC())]) 

# number of trees
grid_param = [{'SVClassifier':[SVC()],
               'SVClassifier__C':[0.2,0.5,1.0,1.5],
               'SVClassifier__gamma':['scale','auto']}]

gridsearch = GridSearchCV(pl_std_svc,grid_param,cv=5,verbose=1,n_jobs=1)
best_model = gridsearch.fit(X_train, y_train)
print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

# saving the best pipeline
with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

#%% 
pl_std_svc = Pipeline([('Standard Scaler',StandardScaler()),
                       ('SVClassifier',SVC(C=1.5,gamma='scale'))]) 
pl_std_svc.fit(X_train,y_train)

# saving the best model
with open(MODEL_PATH,'wb') as file:
  pickle.dump(pl_std_svc,file)

#%% Model Evalution
y_true = y_test
y_pred = pl_std_svc.predict(X_test)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print('Accuracy score: ' + str(round((accuracy_score(y_true, y_pred)*100),3)) + "%")

#%% Conclusion
'''
Since the target column, salary is imbalance,therefore the accuracy score 
which is 91.44% not enough to claim that the SVC model predict the output
accurately. Therefore, f1-score is a better metrics to measure the model 
accuracy in predicting the output. 

For the output B (less than 5k salary),the model can predict the output 
accurately with f1-score of 95%. However, the model accuracy in predicting
the output M (more than 5k salary) is quite low in f1-score, which is 59%. 
Therefore, this model might be biased to the output B in predicting the 
salary group.

A method that can be used to improve the result is by using SMOTE method 
which to oversample the minority class, which is M. Another method that can
be used in predicting the salary group is by using neural networks since 
they are capable of learning more complex relationships which is more suitable
with this kind of dataset and better than pre-existing machine learning 
algorithms. 
'''