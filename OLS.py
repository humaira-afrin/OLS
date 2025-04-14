# OLS code
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn import linear_model
import seaborn as sns
 
df=pd.read_csv("data.csv")
print (df)
df.head()
df1=df.dropna(subset=[ 'BIRTH_WEIGHT_LBS','ATTENTION_R'])


reg=linear_model.LinearRegression()
reg.fit(df1[['age', 'sex', 'gaming_tme', 'ses','sites','BIRTH_WEIGHT_LBS' ]], df1.Whole_brain_Vol)


reg.coef_ 
reg.intercept_
  
Preterm = df1[df1['preterm_birth']==0]
Sex = Preterm[Preterm['sex']==1]
 
X = df1[['age', 'sex', 'gaming_tme', 'ses','sites','BIRTH_WEIGHT_LBS']]
X = Preterm[['age', 'sex', 'gaming_tme', 'ses','sites','BIRTH_WEIGHT_LBS']]
X = Sex [['age', 'sex', 'gaming_tme', 'ses','sites','BIRTH_WEIGHT_LBS']]


Y = df1['ATTENTION_R']
Y = Preterm['ATTENTION_R']
Y = Sex['ATTENTION_R']


reg.fit(df1[['age', 'sex', 'gaming_tme', 'ses','sites','BIRTH_WEIGHT_LBS' ]], df1.Whole_brain_Vol)
#This is the code used to investigate the total cohort and the total brain volume.


reg.fit(Preterm[['age', 'sex', 'gaming_tme', 'ses','sites','BIRTH_WEIGHT_LBS' ]], Preterm.ATTENTION_R)
#This is the code used to investigate full-term and preterm birth and attention ability.


reg.fit(Sex[['age', 'sex', 'gaming_tme', 'ses','sites','BIRTH_WEIGHT_LBS' ]], Sex.ATTENTION_R)
#This is the code used to investigate boys and girls and attention ability


model = sm.OLS(Y,X).fit()
print_model = model.summary()
print(print_model)
