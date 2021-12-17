#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Introduction:
#Problem - Statement: Build an efficient Classification Model that should predict whether it Rains Tomorrow or not, using the dataset Rain in Australia.
#Data source: Kaggle


# In[2]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


pd.set_option("display.max_columns",None)


# In[5]:


#Import Data Set
rain = pd.read_csv('weatherAUS.csv')


# In[6]:


rain.head()


# In[7]:


#Data Preprocessing
rain.shape


# In[8]:


rain.info()


# In[9]:


#categorical data:
categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']
print("Number of Categorical Features: {}".format(len(categorical_features)))
print("Categorical Features: ",categorical_features)


# In[10]:


#Numerical Features:

numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']

#rain.select_dtypes(include=['float64','int64']).columns 

print("Number of Numerical Features: {}".format(len(numerical_features)))
print("Numerical Features: ",numerical_features)


# In[ ]:


Cardinality check for Categorical Features:`

Some Machine Learning algorithms e.g. Logistic Regression, Support Vector Machine can not handle categorical variables and expect all variables to be numeric.
So, categorical data needs to be encoded to numerical data. Before encoding, we need to make sure that categorical features has minimum cardinality.
Cardinality: The number of unique values in feature column is known as cardinality. Example: A column with hundreds of zip codes is an example of a high cardinality feature
A high number of unique values within a feature column is known as high cardinality.
High cardinality may pose some serious problems in the machine learning model.
If a feature column as high cardinality, when we use encoding techniques, then that may cause a significant increase number of dimensions, which is not a good for machine learning problems.
If there is high cardinality in feature column, then:

1) Employ Feature Engineering to extract new features from the feature which possess high cardinality. (or)

2) Simply drop the feature, if that feature doesn't add value to model.


# In[11]:


#Finding cardinality of categorical features:

for each_feature in categorical_features:
    print("Cardinality(no. of unique values) of {} are: {}".format(each_feature,len(rain[each_feature].unique())))


# In[ ]:


Date column has high cardinality which poses several problems to ml model in terms of efficency and also dimenionality of data also increases when converted to numerical data.

Feature enginerring of Date column to decrease high cardinality.


# In[13]:


#Type conversion of Date Column to datetime type:
rain['Date'] = pd.to_datetime(rain['Date'])


# In[14]:


rain['Date'].dtype


# In[15]:


rain['year'] = rain['Date'].dt.year
rain['month'] = rain['Date'].dt.month
rain['day'] = rain['Date'].dt.day


# In[16]:


rain.drop('Date', axis = 1, inplace = True)


# In[17]:


rain.head()


# In[ ]:


` Null (or) NaN (or) Missing Values in Data:`

The real-world data often has a lot of missing(or null) values. The cause of missing values can be data corruption or failure to record data. The handling of missing data is very important during the preprocessing of the dataset as many machine learning algorithms do not support missing values.


# In[18]:


#Handling Null values in categorical features:`
#categorical data: 

categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']
print("Number of Categorical Features: {}".format(len(categorical_features)))
print("Categorical Features: ",categorical_features)


# In[19]:


#Numerical Features:

numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']
#rain.select_dtypes(include=['float64','int64']).columns 
print("Number of Numerical Features: {}".format(len(numerical_features)))
print("Numerical Features: ",numerical_features)


# In[20]:


#Checking for Null values:
rain[categorical_features].isnull().sum()


# In[21]:


# list of categorical features which has null values:

categorical_features_with_null = [feature for feature in categorical_features if rain[feature].isnull().sum()]


# In[22]:


#Filling the missing(Null) categorical features with most frequent value(mode)
for each_feature in categorical_features_with_null:
    mode_val = rain[each_feature].mode()[0]
    rain[each_feature].fillna(mode_val,inplace=True)


# In[23]:


rain[categorical_features].isnull().sum()


# In[24]:


#Handling Null values in numerical features:`
#Checking null values in numerical features
rain[numerical_features].isnull().sum()


# In[25]:


plt.figure(figsize=(15,10))
sns.heatmap(rain[numerical_features].isnull(),linecolor='white')


# In[26]:


# visualizing the Null values in Numerical Features:
rain[numerical_features].isnull().sum().sort_values(ascending = False).plot(kind = 'bar')


# In[ ]:


Null values in Numerical Features can be imputed using Mean and Median. Mean is sensitive to outliers and median is immune to outliers. If you want to impute the null values with mean values, then outliers in numerical features need to be addressed properly.


# In[27]:


#Checking for outliers using Box Plot:
for feature in numerical_features:
    plt.figure(figsize=(10,10))
    sns.boxplot(rain[feature])
    plt.title(feature)
    


# In[28]:


#Checking for outliers using the statistical formulas:
rain[numerical_features].describe()


# In[37]:


#Outlier Treatment to remove outliers from Numerical Features:
# features which have outliers:
features_with_outliers = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','Humidity9am', 'Pressure9am', 'year','month','day']


# In[38]:


#Replacing outliers using IQR:
for feature in features_with_outliers:
    q1 = rain[feature].quantile(0.25)
    q3 = rain[feature].quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (IQR*1.5)
    upper_limit = q3 + (IQR*1.5)
    rain.loc[rain[feature]<lower_limit,feature] = lower_limit
    rain.loc[rain[feature]>upper_limit,feature] = upper_limit


# In[39]:


for feature in numerical_features:
    plt.figure(figsize=(10,10))
    sns.boxplot(rain[feature])
    plt.title(feature)


# In[40]:


#Imputing null values in numerical features using mean:
# list of numerical Features with Null values:
numerical_features_with_null = [feature for feature in numerical_features if rain[feature].isnull().sum()]
numerical_features_with_null


# In[41]:


#Filling null values uisng mean: 
for feature in numerical_features_with_null:
    mean_value = rain[feature].mean()
    rain[feature].fillna(mean_value,inplace=True)


# In[42]:


rain.isnull().sum()


# In[43]:


rain.head()


# In[44]:


#Univariate Analysis`

#Exploring RainTomorrow label
rain['RainTomorrow'].value_counts().plot(kind='bar')


# In[ ]:


Looks like Target variable is imbalanced. It has more 'No' values. If data is imbalanced, then it might decrease performance of model. As this data is released by the meteorological department of Australia, it doesn't make any sense when we try to balance target variable, because the truthfullness of data might descreases. So, let me keep it as it is.


# In[45]:


#Exploring RainToday Variable:
sns.countplot(data=rain, x="RainToday")
plt.grid(linewidth = 0.5)
plt.show()


# In[46]:


#Multivariate Analysis`
plt.figure(figsize=(20,10))
ax = sns.countplot(x="Location", hue="RainTomorrow", data=rain)


# In[47]:


sns.lineplot(data=rain,x='Sunshine',y='Rainfall',color='goldenrod')


# In[48]:


num_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
rain[num_features].hist(bins=10,figsize=(20,20))


# In[49]:


#Correlation:`

#Correlation is a statistic that helps to measure the strength of relationship between features. 
rain.corr()


# In[50]:


plt.figure(figsize=(20,20))
sns.heatmap(rain.corr(),linewidths=0.5,annot=True,fmt=".2f")


# In[51]:


rain.head()


# In[ ]:


`Encoding of Categorical Features`

Most Machine Learning Algorithms like Logistic Regression, Support Vector Machines, K Nearest Neighbours etc. can't handle categorical data. Hence, these categorical data needs to converted to numerical data for modelling, which is called as Feature Encoding.


# In[52]:


categorical_features


# In[53]:


#Encoding Categorical Features using replace function:
rain['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)
#pd.get_dummies(rain['RainToday'],drop_first = True)
rain['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)


# In[54]:


def encode_data(feature_name):
  
     #function which takes feature name as a parameter and return mapping dictionary to replace(or map) categorical data to numerical data.
    
    mapping_dict = {}
    unique_values = list(rain[feature_name].unique())
    for idx in range(len(unique_values)):
        mapping_dict[unique_values[idx]] = idx
    print(mapping_dict)
    return mapping_dict


# In[55]:


rain['WindGustDir'].replace(encode_data('WindGustDir'),inplace = True)


# In[56]:


rain['WindDir9am'].replace(encode_data('WindDir9am'),inplace = True)


# In[57]:


rain['WindDir3pm'].replace(encode_data('WindDir3pm'),inplace = True)


# In[58]:


rain['Location'].replace(encode_data('Location'), inplace = True)


# In[59]:


rain.head()


# In[60]:


#Spliting data into input features and label
X = rain.drop(['RainTomorrow'],axis=1)
y = rain['RainTomorrow']


# In[ ]:


`Feature Importance:`

- Machine Learning Model performance depends on features that are used to train a model. 
- Feature importance describes which features are relevant to build a model. 
- Feature Importance refers to the techniques that assign a score to input/label features based on how useful they are  at predicting a target variable. Feature importance helps in Feature Selection.


# In[63]:


#finding feature importance using ExtraTreesRegressor:
from sklearn.ensemble import ExtraTreesRegressor
etr_model = ExtraTreesRegressor()
etr_model.fit(X,y)


# In[64]:


ExtraTreesRegressor()


# In[65]:


etr_model.feature_importances_


# In[66]:


# visualizing feature importance using bar graph:
feature_imp = pd.Series(etr_model.feature_importances_,index=X.columns)
feature_imp.nlargest(10).plot(kind='barh')


# In[67]:


feature_imp


# In[ ]:


Split Data into Training and Testing Set


# In[68]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[69]:


print("Length of Training Data: {}".format(len(X_train)))
print("Length of Testing Data: {}".format(len(X_test)))


# In[ ]:


Feature Scaling


# In[70]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# In[71]:


X_test = scaler.transform(X_test)


# In[ ]:


Save the Scaler object to Standardize Real Time Data feeded by users for prediction


# In[72]:


with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


# In[ ]:


Model Building
- Model Training
- Model Testing 
- Evaluating Model Performance using Accuracy, Confusion Matrix, Classification Report, RUC-AUC curve
- Finding whether model performance can be improved using Cross Validation Score


# In[73]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[74]:


#`(i) Logistic Regression`

#Model Training:
from sklearn.linear_model import LogisticRegression


# In[75]:


start_time = time.time()
classifier_logreg = LogisticRegression(solver='liblinear', random_state=0)
classifier_logreg.fit(X_train, y_train)
end_time = time.time()


# In[76]:


print("Time Taken to train: {}".format(end_time - start_time))


# In[77]:


#Model Testing:
y_pred = classifier_logreg.predict(X_test)


# In[78]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[79]:


print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred)))


# In[80]:


#Checking for Overfitting and Under Fitting:
print("Train Data Score: {}".format(classifier_logreg.score(X_train, y_train)))
print("Test Data Score: {}".format(classifier_logreg.score(X_test, y_test)))


# In[ ]:


Accuracy Score of Training and Testing Data is comparable and almost equal. So, there is no question of Underfitting and Over Fitting. And model is generalizing well for new unseen data.


# In[81]:


#confusion Matrix:
print("Confusion Matrix:")
print("\n",confusion_matrix(y_test,y_pred))


# In[82]:


#Classification Report:
print("classification_report:")
print("\n",classification_report(y_test,y_pred))


# In[83]:


#predicting probabilities:
y_pred_logreg_proba = classifier_logreg.predict_proba(X_test)


# In[84]:


#Finding True Positive Rate(tpr), False Positive Rate(fpr), threshold values to plot ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg_proba[:,1])


# In[85]:


#Plotting ROC curve:
plt.figure(figsize=(6,4))
plt.plot(fpr,tpr,'-g',linewidth=1)
plt.plot([0,1], [0,1], 'k--' )
plt.title('ROC curve for Logistic Regression Model')
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.show()


# In[86]:


#finding ROC-AUC score:
from sklearn.metrics import roc_auc_score
print('ROC AUC Scores: {}'.format(roc_auc_score(y_test, y_pred)))


# In[ ]:


Finding whether model performance can be improved using Cross Validation Score:


# In[87]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier_logreg, X_train, y_train, cv = 5, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))


# In[88]:


print('Average cross-validation score: {}'.format(scores.mean()))


# In[ ]:


The mean accuracy score of cross validation is almost same like original model accuracy score which is 0.8445. So, accuracy of model may not be improved using Cross-validation.


# In[92]:


#`(ii) Cat Boost`

from catboost import CatBoostClassifier


# In[93]:


#Model Training:

start_time = time.time()
cat_classifier = CatBoostClassifier(iterations=2000, eval_metric = "AUC")
cat_classifier.fit(X_train, y_train)
end_time = time.time()


# In[94]:


print("Time Taken to train: {}".format(end_time - start_time))


# In[95]:


#Model Testing:

y_pred_cat = cat_classifier.predict(X_test)


# In[96]:


print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred_cat)))


# In[97]:


#Checking for Overfitting and Under Fitting:
print("Train Data Score: {}".format(cat_classifier.score(X_train, y_train)))
print("Test Data Score: {}".format(cat_classifier.score(X_test, y_test)))


# In[ ]:


Accuracy Score of Training and Testing Data is comparable and almost equal. So, there is no question of Underfitting and Over Fitting. And model is generalizing well for new unseen data.


# In[100]:


#Confusion Matrix:
print("Confusion Matrix:")
print("\n",confusion_matrix(y_test,y_pred_cat))


# In[101]:


#classification Report:
print("classification_report:")
print("\n",classification_report(y_test,y_pred_cat))


# In[102]:


#predicting the probabilities:
y_pred_cat_proba = cat_classifier.predict_proba(X_test)


# In[103]:


#Finding True Positive Rate(tpr), False Positive Rate(fpr), threshold values to plot ROC curve  
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_cat_proba[:,1])


# In[104]:


#plotting ROC Curve:
plt.figure(figsize=(6,4))
plt.plot(fpr,tpr,'-g',linewidth=1)
plt.plot([0,1], [0,1], 'k--' )
plt.title('ROC curve for Cat Boost Model')
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.show()


# In[105]:


#finding ROC AUC Scores:
from sklearn.metrics import roc_auc_score
print('ROC AUC Scores: {}'.format(roc_auc_score(y_test, y_pred_cat)))


# In[106]:


#Finding whether model performance can be improved using Cross Validation Score:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(cat_classifier, X_train, y_train, cv = 5, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))


# In[108]:


print('Average cross-validation score: {}'.format(scores.mean()))


# In[ ]:


Average cross-validation score: 0.8632871444807307
The mean accuracy score of cross validation is almost same like original model accuracy score which is 0.8647050735597415. So, accuracy of model may not be improved using Cross-validation.


# In[109]:


#`(iii) Random Forest`

from sklearn.ensemble import RandomForestClassifier


# In[110]:


#Model Training:

start_time = time.time()
classifier_rf=RandomForestClassifier()
classifier_rf.fit(X_train,y_train)
end_time = time.time()


# In[111]:


print("Time Taken to train: {}".format(end_time - start_time))


# In[112]:


#Model Testing:
y_pred_rf = classifier_rf.predict(X_test)


# In[113]:


print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred_rf)))


# In[114]:


#Checking for Overfitting and Under Fitting:
print("Train Data Score: {}".format(classifier_rf.score(X_train, y_train)))
print("Test Data Score: {}".format(classifier_rf.score(X_test, y_test)))


# In[ ]:


Accuracy score for Training Set is almost 1 or 100%, which is quite uncommon. And testing accuracy is 0.85. It seems like model is overfitting, because the generalization for unseen data is not that accurate, when compared with seen data and difference between training - testing accuracy is not minimum.


# In[ ]:


Results and Conclusion:
Best Models in terms of accuracy (In my Experiment):

1) Cat Boost Model
2) Logistic Regression
3) Random Forest

Best Models in terms of Computation Time (In my Experiment):

1) Logistic Regression
2) Random Forest
3) Cat Boost Model
Conclusion:

The accuracy score of Cat Boost Model is high when compared with accuracy scores of Logistic Regression and Random Forest. But cat Boost model consumes lot of time to train the model.

In terms of computation time and Accuracy score, logistic Regression model is doing job.


# In[ ]:


Saving Classifier Object into Pickle File:


# In[115]:


with open('logreg.pkl', 'wb') as file:
    pickle.dump(classifier_logreg, file)


# In[116]:


with open('catboostclassifier.pkl', 'wb') as file:
    pickle.dump(cat_classifier, file)

