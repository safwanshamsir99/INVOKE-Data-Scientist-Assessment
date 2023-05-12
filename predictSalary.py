# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 18:46:02 2023

@author: Safwan
"""
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the test data and ml model
TEST_CSV = os.path.join(os.getcwd(),'test_data.csv')
test_data = pd.read_csv(TEST_CSV)
MODEL_PATH = os.path.join(os.getcwd(),'best_model_ft.pkl')
with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

def predictSalary(model=model,data=test_data):
    # Select only the required features
    features = ['age','house_rental_fee','house_loan_pmt','transport_spending',
                'public_transport_spending','house_utility','food_spending',
                'kids_spending','personal_loan','other_loan','investment']
    X_test = test_data[features]
    
    # Impute missing values in X_test
    ii_imputer = IterativeImputer()
    X_test = pd.DataFrame(ii_imputer.fit_transform(X_test), columns=X_test.columns)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # convert the predictions to integer labels
    y_pred = np.round(y_pred).astype(int)

    # Inverse transform the encoded predictions
    ENCODER_PATH = os.path.join(os.getcwd(),'le_salary.pkl')
    with open(ENCODER_PATH, 'rb') as f:
        decoder = pickle.load(f)
    y_pred = decoder.inverse_transform(y_pred)
    df_output = pd.DataFrame(y_pred)
    
    # Save the dataframe output into csv file
    df_output.to_csv('output.csv',index=False)
    
    return df_output

predictSalary(model=model, data=test_data)
