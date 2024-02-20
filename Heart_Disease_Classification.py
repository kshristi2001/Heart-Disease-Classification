#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import random
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score,classification_report,f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import itertools
from sklearn import metrics


# In[2]:


get_ipython().system('pip install xgboost')


# In[3]:


# We are using cleaned_data.csv dataset that we obtained and printing first 10 fields
df = pd.read_csv('heart.csv')
df.head(10)


# In[4]:


#printing the shape of the dataset
df.shape


# In[5]:


#column lables of the dataframe
df.columns


# In[6]:


#data trpes of attributes in the dataset
df.dtypes


# In[7]:


# Checking the shape of the data
print("rows:",df.shape[0],"columns:",df.shape[1])


# In[8]:


#Removing all the leading and trailing spaces in the data using strip() function
df.columns = df.columns.str.strip();
df.head(406)


# In[9]:


#information of the dataframe
df.info()


# In[10]:


#Return a Series containing counts of unique rows in the DataFrame.
df['target'].value_counts()


# In[11]:


#Checking for the duplicate values and removing them
dups = df.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
df=df.drop_duplicates()


# In[12]:


# Double checking if there are any duplicate values
dups = df.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))


# In[13]:


#Checking the descriptive statistics of the dataframe
df.describe()


# In[14]:


#Displaying the null values
df.isnull().sum()


# In[15]:


#Drawing boxplot to identify the features/columns that contain outliers
fig, ax = plt.subplots(figsize=(10,5))  
df.boxplot(ax=ax)

#Adding the title and xlabel 
ax.set_title('Boxplot of Data Columns')
ax.set_xlabel('Column')

#Displaying the plot
plt.show()


# In[16]:


#Drawing the boxplots which only contains the outliers
df[['resting_bp', 'cholestoral','fasting_blood_sugar', 'max_hr','oldpeak', 'num_major_vessels', 'thal']].plot.box(figsize = (10, 6))


# In[17]:


#plotting a pie graph for heart disease, and no heart disease
labels = "heart disease", "no heart disease"
explode = (0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(df.target.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

disease = len(df[df.target == 0])
non_disease = len(df[df.target == 1])

#printing the percentage of the people with and without the heart disease 
print("Percentage of the People without Heart Disease : {:.2f}%".format((non_disease / (len(df.target))*100)))
print("Percentage of the People with Heart Disease: {:.2f}%".format((disease / (len(df.target))*100)))


# In[18]:


#plotting a bar chart for female and male patients
sns.countplot(x='sex', data=df, palette="RdGy")
plt.xlabel("female = 0 and male = 1")
plt.show()

female = len(df[df.sex == 0])
male = len(df[df.sex == 1])

#printing the percentage of female and male patients
print("Percentage of Female Patients: {:.2f}%".format((female / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((male / (len(df.sex))*100)))


# In[19]:


#plotting a bar chart for frequency of heart disease against age
color = [random.choice(list(cm.colors.CSS4_COLORS.values())) for i in range(len(df.target.unique()))]
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6), legend=False, color = color)
plt.title('frequency of heart disease for age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[20]:


#plotting a scatter plot against the distribution of heart rate over the age
sns.scatterplot(data=df, x='age', y='max_hr', hue='target', palette='PuRd')
plt.legend(["Disease", "Non-Disease"])
plt.title("distribution of heart rate over the age",fontsize=15)
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[21]:


#Drawing multiple histograms for all numerical attributes
df[['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholestoral',
       'fasting_blood_sugar', 'restecg', 'max_hr', 'exang', 'oldpeak', 'slope',
       'num_major_vessels', 'thal', 'target']].hist(figsize=(10, 10), bins=10, alpha = 0.75, color = 'y')
plt.tight_layout()
plt.show()


# In[22]:


#scatter plot for distribution of cholestoral over the age 
plt.figure(figsize=(10,10))
sns.relplot(data=df,x="age",y="cholestoral",hue="sex")#Not relplot is used to find the distribution of cholestroal over target
plt.xlabel("Age",fontsize=15)# Note Labelling the x axis
plt.ylabel("cholestoral",fontsize=15)# Note Labelling the y axis
plt.title("distribution of cholestoral over the age",fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[23]:


#plotting a face grid for blood pressure as per the gender
plt.figure(figsize=(20,10))
d=sns.FacetGrid(data=df,hue="sex",aspect=3)# Note it takes the data as input and variables as rows, column
d.map(sns.kdeplot,"resting_bp",shade=True)#Mapping of the data  is done and kdeplot is used
plt.ylabel("Count",fontsize=15)# Note Labelling the y axis
plt.xlabel("blood pressure",fontsize=15)# Note Labelling the x axis
plt.title("Blood Pressure as per the Gender",fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(labels=["Female","Male"])
plt.show()


# In[24]:


# Correlation matrix is used to know the dependancy between the variables in the data
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),square=True,annot=True,linewidths=0.5)
plt.title('Corelation Between Variables',fontsize=20)
plt.show()


# In[25]:


#Creating a scatterplot matrix for all the features
axes = scatter_matrix(df, alpha=0.2, figsize = (25, 25))


# In[26]:


#plotting a bar chart for heart disease frequency according to fasting blood sugar
my_colors = 'bk'
pd.crosstab(df.fasting_blood_sugar,df.target).plot(kind="bar",figsize=(10,5), color=my_colors)
plt.title('Heart Disease Frequency According To Fasting Blood Sugar')
plt.xlabel('Fasting Blood Sugar greater than 120 mg/dl (1=True and 0=false)')
plt.xticks(rotation = 0)
plt.legend(["Non-Disease", "Disease"])
plt.ylabel('Frequency of having heart Disease or not')
plt.show()


# In[27]:


#plotting a bar chart for heart disease frequency according to chest pain type
my_colors = 'gr'
pd.crosstab(df.chest_pain_type,df.target).plot(kind="bar",color=my_colors, figsize=(10,5))
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.legend(["Non-Disease", "Disease"])
plt.xticks(rotation = 0)
plt.ylabel('Frequency of having heart Disease or not')
plt.show()


# In[28]:


#plotting a bar chart for heart disease frequency according to the slope
my_colors='cm'
pd.crosstab(df.slope,df.target).plot(kind="bar", color=my_colors, figsize=(10,5))
plt.title('Heart Disease Frequency according to Slope')
plt.legend(["Non-Disease", "Disease"])
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of having heart Disease or not')
plt.show()


# In[29]:


#splitting the dataset into test and train and fitting it
y = df['target']
X = df.drop('target',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[30]:


#printing the shape of test and train data
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[31]:


#classifying using logestic regression
logestic_regg = LogisticRegression()
logestic_regg.fit(X_train,y_train)

#prediction using logestic regression
y_pred_log_lr = logestic_regg.predict(X_test)


# In[32]:


#finding the accuracy, F1 score, confusion matrix and classification report for logistic regression
f1_lr = f1_score(y_test, y_pred_log_lr)
accuracy_lr = accuracy_score(y_test, y_pred_log_lr)
confusion_mat_lr = confusion_matrix(y_test, y_pred_log_lr)
classification_rep_lr = classification_report(y_test, y_pred_log_lr)


# In[33]:


#printing accuracy, F1 score, confusion matrix and classification report for logistic regression
print("F1 Score: ", f1_lr)
print("Accuracy: ", accuracy_lr)
print("Confusion Matrix:\n", confusion_mat_lr)
print("Classification Report:\n", classification_rep_lr)


# In[34]:


#classification using KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)


# In[35]:


#finding the accuracy, F1 score, confusion matrix and classification report for KNN classifier
f1_knn = f1_score(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
confusion_mat_knn = confusion_matrix(y_test, y_pred_knn)
classification_rep_knn = classification_report(y_test, y_pred_knn)


# In[36]:


#printing the accuracy, F1 score, confusion matrix and classification report for KNN classifier
print("F1 Score: ", f1_knn)
print("Accuracy: ", accuracy_knn)
print("Confusion Matrix:\n", confusion_mat_knn)
print("Classification Report:\n", classification_rep_knn)


# In[37]:


# classification using extreme gradient boosting
xgb = XGBClassifier(use_label_encoder=False,eval_metric='error')
xgb.fit(X_train,y_train)

y_pred_xgb = xgb.predict(X_test)


# In[38]:


#finding the accuracy, F1 score, confusion matrix and classification report for extreme gradient boosting classifier
f1_xgb = f1_score(y_test, y_pred_xgb)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
confusion_mat_xgb = confusion_matrix(y_test, y_pred_xgb)
classification_rep_xgb = classification_report(y_test, y_pred_xgb)


# In[39]:


#printing the accuracy, F1 score, confusion matrix and classification report for extreme gradient boosting classifier
print("F1 Score: ", f1_xgb)
print("Accuracy: ", accuracy_xgb)
print("Confusion Matrix:\n", confusion_mat_xgb)
print("Classification Report:\n", classification_rep_xgb)


# In[40]:


#classification using decision tree classifier
des_tree = DecisionTreeClassifier()
des_tree.fit(X_train,y_train)
y_pred_des = des_tree.predict(X_test)


# In[41]:


#finding the accuracy, F1 score, confusion matrix and classification report for decision tree classifier
f1_dt = f1_score(y_test, y_pred_des)
accuracy_dt = accuracy_score(y_test, y_pred_des)
confusion_mat_dt = confusion_matrix(y_test, y_pred_des)
classification_rep_dt = classification_report(y_test, y_pred_des)


# In[42]:


#printing the accuracy, F1 score, confusion matrix and classification report for decision tree classifier
print("F1 Score: ", f1_dt)
print("Accuracy: ", accuracy_dt)
print("Confusion Matrix:\n", confusion_mat_dt)
print("Classification Report:\n", classification_rep_dt)


# In[43]:


#plotting a table for comparing the accuracy and the F1 score of the classifiers
model_grid_df = pd.DataFrame({'Model': ['Logistic Regression','K-Nearest Neighbour','Extreme Gradient Boost',
                              'Decision Tree'],
                              'Accuracy':[accuracy_lr,accuracy_knn,accuracy_xgb,accuracy_dt],
                              'F1 score':[f1_lr, f1_knn ,f1_xgb,f1_dt]})


# In[44]:


#printing the table
model_grid_df


# In[45]:


#plotting the confusion matrices for all the classifiers
plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(confusion_mat_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(confusion_mat_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("XG Bosting Confusion Matrix")
sns.heatmap(confusion_mat_xgb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(confusion_mat_dt,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()


# In[46]:


#plotting the ROC curve for logistic regression, KNN, XGB and decision tree classifiers
y_pred_grid = [y_pred_log_lr,y_pred_knn,y_pred_xgb,y_pred_des]
model_name = ['Logistic Regression','KNN','XGB','Decision tree']

curve = []

for y_pred_grid_ in y_pred_grid:
    curve.append(roc_curve(y_test,y_pred_grid_))


# In[47]:


#plotting the ROC curve for logistic regression, KNN, XGB and decision tree classifiers
plt.plot([0,1],[0,1],'k--')

for i in range(len(model_name)):
    plt.plot(curve[i][0],curve[i][1],label=model_name[i]+" Grid")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.show()


# In[48]:


#plotting a graph for the most important features in the dataset 
feature_importance = abs(logestic_regg.coef_[0])
feature_importance = 100.0 *(feature_importance/feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + 0.5

featfig = plt.figure()
featax = featfig.add_subplot(1,1,1)
featax.barh(pos,feature_importance[sorted_idx],align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx],fontsize=10)
featax.set_xlabel('Relative Feature Importance')

plt.tight_layout()
plt.show()

#variable num_major_vessels was the most influential in predicting whether an individual had heart disease or not.
#Variable num_major_vessels referred the the number of major vessels (0–3) colored by a flourosopy. 
#The lower the number of major blood vessels, reduces the amount of blood flowing to the heart, 
#increasing the presence of heart disease.


# In[ ]:




