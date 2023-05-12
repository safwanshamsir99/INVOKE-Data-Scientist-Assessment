# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:16:56 2023

@author: Acer
"""

import os
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.getcwd(),'surveyA.csv')
df = pd.read_csv(CSV_PATH)

test_size = int(len(df) / 5)
test_df = df.iloc[-test_size:]

# Save the test set to a new CSV file
test_df.to_csv('test_data.csv', index=False)