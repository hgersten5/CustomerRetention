# Databricks notebook source
# DBTITLE 1,Customer Retention Analysis
# MAGIC %md
# MAGIC <b>Objective</b>
# MAGIC 
# MAGIC <br/>I will explore the data and try to analyze the following: 
# MAGIC <br/>What's the % of Churn Customers ?
# MAGIC <br/>Is there any patterns in Churn Customers based on specific services? 
# MAGIC <br/>What's the most profitable service types?
# MAGIC <br/>Which features and services are most profitable?
# MAGIC 
# MAGIC After, I would like to build a customer retention model to try to predict which customers are most likely to churn to be able to retain them
# MAGIC 
# MAGIC (Building a model for customer retention is important because it allows businesses to better understand customer behaviour and needs. A model can help identify customers who are most likely to churn and put in place strategies to retain them. It also helps businesses segment their customers into different categories and target them with relevant campaigns, offers and services. By predicting customer churn, businesses can proactively reach out to customers, offer discounts and incentives, and create personalized experiences. Ultimately, a model for customer retention helps businesses reduce churn, increase profits and build customer loyalty.)

# COMMAND ----------

# MAGIC %md 
# MAGIC <b>Pre-processing the data</b>

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

df =  pd.read_csv('/dbfs/FileStore/tables/Customers.csv')

# COMMAND ----------

df.head()

# COMMAND ----------

#check for missing values
cols = df.columns
cols = cols.drop(['customerID','gender','tenure'])#getting cols for visualazation purposes 
df.info()

# COMMAND ----------

# MAGIC %md 
# MAGIC It looks like there are all non null values. Let's convert our columns to numeric so that we can find the corrolation between itself and the target column

# COMMAND ----------

for col in df.columns[2:]:
    print(df[col].value_counts())

# COMMAND ----------

#df['TotalCharges'] = df['TotalCharges'].replace('',0)
df[df['TotalCharges']==' ']
#It looks like there are a number of records that have no total charges and monthly charges. They also have 0 tenure
#for now,let's filter those out, since I am concerned for their data will sku the analysis
df = df[df['TotalCharges']!=' ']

# COMMAND ----------

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# COMMAND ----------

#the ones that are no yes, we can make into 0,1
cols_to_convert = ['Churn','PaperlessBilling','PhoneService','Dependents','Partner']
cols_to_copy= ['PaymentMethod','Contract','StreamingMovies','StreamingTV','TechSupport','DeviceProtection','OnlineBackup','OnlineSecurity','InternetService','MultipleLines']
joined_cols = cols_to_convert +cols_to_copy

for col in cols_to_convert:
    df[col] = df[col].astype('category').cat.codes
    
for col in cols_to_copy:
    df['_num'+col]=df[col].astype('category').cat.codes
joined_cols = cols_to_convert +cols_to_copy


# COMMAND ----------

#values look consistent. Now we can start checking for corrolation


# COMMAND ----------

# MAGIC %md 
# MAGIC <b>Feature Analysis and Feature Engineering</b>

# COMMAND ----------

df.describe()
#it looks like there is an avg churn rate of 26.5%

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like on average the churn rate is 26.5%. It would be interesting to see what it would be like when filtering out the dataset 
# MAGIC 
# MAGIC It looks like people with phone service and do not elect for the internet service are most likely to churn.
# MAGIC <br>In other words - customers with more services are more likely to stick around
# MAGIC Let's filter the data further and see if we get a higher churn rate 

# COMMAND ----------

#let's try to see the mean churn rate filtering differently
print(df.loc[(df['PhoneService'] ==1)& (df['StreamingMovies']=="No internet service" ) & (df['StreamingTV']=="No internet service" ) ,'Churn'].mean())
print(df.loc[(df['PhoneService'] ==1) & (df['StreamingMovies']=="No internet service" ) & (df['StreamingTV']=="No internet service" ) ,'Churn'].count())
print(df.loc[(df['PhoneService'] ==1) & (df['MultipleLines']=="No" ) & (df['StreamingMovies']=="No" ) & (df['StreamingTV']=="No" ) & (df['InternetService']=="Fiber optic" ) & (df['DeviceProtection']=="No" ) & (df['OnlineBackup']=="No" ) & (df['OnlineSecurity']=="No" ) & (df['TechSupport']=="No" ),'Churn'].mean())
print(df.loc[(df['PhoneService'] ==1) & (df['MultipleLines']=="No" ) & (df['StreamingMovies']=="No" ) & (df['StreamingTV']=="No" ) & (df['InternetService']=="Fiber optic" ) & (df['DeviceProtection']=="No" ) & (df['OnlineBackup']=="No" ) & (df['OnlineSecurity']=="No" ) & (df['TechSupport']=="No" ),'Churn'].count())
print(df.loc[(df['PhoneService'] ==0) & (df['InternetService']!="No" ),'Churn'].count())

# COMMAND ----------

# MAGIC %md 
# MAGIC It looks like:
# MAGIC <br >--the more services you have the less likely you are to churn. Let's add 
# MAGIC <br >--have phone service, and no internet service have a low churn rate 
# MAGIC <br> let's add num service and visualize it to confirm

# COMMAND ----------

#let's filter out to users that have internet service because the other have a very little churn rate
service_df = df
#service_df= df[df['InternetService']!="No"]
service_cols = ['PhoneService','_numMultipleLines','_numStreamingMovies','_numStreamingTV','_numDeviceProtection','_numOnlineBackup','_numOnlineSecurity','_numTechSupport']
for col in service_cols: 
    service_df.loc[service_df[col] ==1, col] = 0
    service_df.loc[service_df[col] ==2, col] = 1
    
df['NumServices'] =service_df.loc[:,service_cols].sum(axis=1) 
df['NumServices'].value_counts()

# COMMAND ----------

categorical_labels = [['PaymentMethod', 'Contract']
                     ,['PhoneService','MultipleLines'],['StreamingMovies','StreamingTV'],['DeviceProtection','InternetService'],['OnlineBackup','OnlineSecurity'],['NumServices','SeniorCitizen']]

fig, ax = plt.subplots(6, 2, figsize=(25, 25))
for i in range(6):
    for j in range(2):
        feature = categorical_labels[i][j]
        ax1 = sns.countplot(x=feature, hue='Churn', palette=["#FAAE7B",'#432371'], data=df, ax=ax[i][j])
        ax1.set_xlabel(feature, labelpad=10)
        ax1.set_ylim(0, 6000)
        ax1.legend(title='Churn', labels= ['No', 'Yes'])
     #   if i == 1:
       #     ax1.set_xticklabels(['No', 'Yes'])
sns.despine()


# COMMAND ----------

# MAGIC %md
# MAGIC yes - the more services you have the lower the churn rate

# COMMAND ----------

#It looks like all the services are good features, as well as Tenure.
features =['_numPaymentMethod', '_numContract','PhoneService','_numMultipleLines','_numStreamingMovies','_numStreamingTV','_numDeviceProtection','_numInternetService','_numOnlineBackup','_numOnlineSecurity','NumServices']

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's look for corralation between columns

# COMMAND ----------

corr_matrix = df.corr()
print(corr_matrix["Churn"].sort_values(ascending=False))


# COMMAND ----------

# MAGIC %md 
# MAGIC It looks like tenure and num contract columns may have potential.
# MAGIC Let's try to bin the tenure column

# COMMAND ----------

pd.qcut(df['tenure'], q=4)


# COMMAND ----------

df['tenure'].hist(bins=[-1,0,9,30,56,72])

# COMMAND ----------

sorted(df['tenure'].unique())
#it looks like it ranges from 0-72
bins = [-1,0,9,30,56,72]
labels = [1,2,3,4,5]
df['TenureRange'] = pd.cut(df['tenure'], bins=bins, labels=labels)
df['TenureRange'] = df['TenureRange'].astype(int)

# COMMAND ----------

corr_matrix = df.corr()
print(corr_matrix["Churn"].sort_values(ascending=False))

# COMMAND ----------

# MAGIC %md 
# MAGIC Yes! The TenureRange and tenure now seem to have a higher core.

# COMMAND ----------

features.append('TenureRange')

# COMMAND ----------

# MAGIC %md
# MAGIC <b>Investigate the best model</b>

# COMMAND ----------

from sklearn.feature_selection import RFECV
df_num = df.select_dtypes([np.number]).dropna(axis=1)
all_X = df_num.drop(columns=['Churn'],axis=1)
all_y = df_num['Churn']
rf = RandomForestClassifier(random_state=1)
rfecv = RFECV(estimator=rf, cv=10)
rfecv.fit(all_X, all_y)
X_new = rfecv.transform(all_X)
print("Num Features Before:", all_X.shape[1])
print("Num Features After:", X_new.shape[1])
features_kept = pd.DataFrame({'columns': all_X.columns,
                                 'Kept': rfecv.support_})

features = list(all_X.columns)


# COMMAND ----------

features_kept[columns]
#I want to keep the columns even if false 

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

df1 = df[features].select_dtypes([np.number]).dropna(axis=1)
all_X = df1[features]
all_y = df["Churn"]
models = [
            {"model":"LogisticRegression",
                  "estimator":LogisticRegression(),
                  "parameters": {'solver': ['newton-cg','lbfgs','liblinear']}
            },
             {
                 
             "model":"KNeighborsClassifier",
             "estimator":KNeighborsClassifier(),
             "parameters": {'n_neighbors':range(1,20,2),
                                          'weights': ['distance','uniform'],
                                          'algorithm': ['ball_tree','kd_tree','brute'],
                                          'p': [1,2]
                                          }
             },
            {
               "model":"RandomForestClassifier",
               "estimator":RandomForestClassifier(),
               "parameters":{'n_estimators':[4,6,9],
                                           'criterion': ['entropy','gini'],
                                           'max_depth': [2,5,10],
                                           'max_features': ['log2','sqrt'],
                                           'min_samples_leaf':[1,5,8],
                                           'min_samples_split':[2,3,5]
                                           }
            }
    ]
for model in models:
            print(model['model'])
            print('-'*len(model['model']))

            grid = GridSearchCV(model["estimator"],
                            param_grid=model["parameters"],
                            cv=10)
            grid.fit(all_X,all_y)
            model["best_params"] = grid.best_params_
            model["best_score"] = grid.best_score_
            model["best_model"] = grid.best_estimator_

            print("Best Score: {}".format(model["best_score"]))
            print("Best Parameters: {}\n".format(model["best_params"]))

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like the RandomForestClassifier predicted the best.

# COMMAND ----------

chosen_model =model['best_model']

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier 
X_train, X_test, y_train, y_test = train_test_split(df1[features], df["Churn"], test_size=0.2, random_state=0)

#train the model
chosen_model.fit(X_train, y_train)

#predict on test set
y_pred = chosen_model.predict(X_test)

#check accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# COMMAND ----------


