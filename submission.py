import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
data_dir = "/Users/akshay/Desktop/A4/DT_data/"
os.listdir(data_dir)

train_data = pd.read_csv(data_dir+"train.csv")
x1 = pd.get_dummies(train_data[" Work Class"],prefix_sep="_")
x2 = pd.get_dummies(train_data[" Marital Status"],prefix_sep="_")
x3 = pd.get_dummies(train_data[" Occupation"],prefix_sep="_")
x4 = pd.get_dummies(train_data[" Relationship"],prefix_sep="_")
x5 = pd.get_dummies(train_data[" Race"],prefix_sep="_")
x6 = pd.get_dummies(train_data[" Native Country"],prefix_sep="_")
# x7 = pd.get_dummies(train_data[" Education"],prefix_sep="_")
X = pd.concat([x1,
               x2,
               x3,
               x4,
               x5,
               x6,
               train_data["Age"],
#                train_data[" Fnlwgt"],
               train_data[" Education Number"],
               train_data[" Capital Gain"],
               train_data[" Capital Gain"],
               train_data[" Capital Loss"],
               train_data[" Hour per Week"],
               train_data[" Rich?"]],axis=1)
train = X.iloc[:20000,:]
test = X.iloc[20000:,:]