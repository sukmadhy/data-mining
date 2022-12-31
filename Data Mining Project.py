#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[107]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ## import dataset

# In[108]:


df_titanic = pd.read_csv('Titanic.tsv', sep = '\t')


# In[109]:


df_titanic.info()


# In[110]:


df_titanic.tail()


# ## Find duplicates

# In[111]:


df_titanic['PassengerId'].duplicated().sum()


# In[112]:


df_titanic[df_titanic['PassengerId'].duplicated()]


# we found the duplicates, thus we need to remove the duplicates.

# In[113]:


df_titanic = df_titanic.drop_duplicates(subset=['PassengerId'], keep='first')


# In[114]:


df_titanic['PassengerId'].duplicated().sum()


# In[115]:


df_titanic.info()


# ## repairing the dataset (manipulate the value) and change the data type

# There are inconsistent values in the some columns. Hence, we need to change them to proper values.

# ### PassengerId

# I will change the data type of PassengerId because we won't do aggregation to this column

# In[116]:


#change the data type

df_titanic['PassengerId'] = df_titanic['PassengerId'].astype(str)
df_titanic.info()


# In[117]:


df_titanic['PassengerId'].value_counts()


# ### Name

# In[118]:


df_titanic['Name'].value_counts()


# In[119]:


df_titanic['Name'].duplicated().sum()


# In[120]:


df_titanic[df_titanic['Name'].duplicated()]


# In[121]:


df_titanic = df_titanic.drop_duplicates(subset=['Name'], keep='first')
df_titanic['Name'].value_counts()


# In[122]:


df_titanic['Name'].duplicated().sum()


# In[123]:


df_titanic['Name'] = df_titanic['Name'].replace('Mr. Frederick Maxfield Hoyt','Hoyt, Mr. Frederick Maxfield')


# In[124]:


df_titanic['FirstName'] = df_titanic['Name'].str.split(',').str[1]
df_titanic['LastName'] = df_titanic['Name'].str.split(',').str[0]
df_titanic.tail(5)


# In[125]:


df_titanic['FirstName'].value_counts()


# In[126]:


df_titanic['FirstName'] = df_titanic['FirstName'].str.split('.').str[1]
df_titanic.tail(5)


# In[127]:


df_titanic = df_titanic[["PassengerId","Survived","Pclass","FirstName","LastName","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked","ship"]]
df_titanic.tail()


# ### Survived

# In[128]:


df_titanic["Survived"].value_counts()


# In[129]:


df_titanic["Survived"].mode()


# We will change -4 to 0 because 0 is the nearest value to -4 and also the most frequent values.

# In[130]:


df_titanic['Survived'] = df_titanic['Survived'].replace(-4, 0)
df_titanic['Survived'].value_counts()


# ### Sex

# In[131]:


df_titanic["Sex"].value_counts()


# As we can see, there are inconsistency in the column of gender. In this column we will change it to only two labels. Female and Male.

# In[132]:


#change the label

# male
df_titanic['Sex'] = df_titanic['Sex'].replace(['malef', 'mal', 'malee', 'male'], 'Male')

# female
df_titanic['Sex'] = df_titanic['Sex'].replace(['fem', 'femmale', 'feemale', 'Female','F', 'female'], 'Female')

df_titanic['Sex'].value_counts()


# ### Age

# There are some values of age that has comma and are not even numbers, so we will change it to period and even numbers.

# In[133]:


df_titanic['Age'].value_counts()


# In[134]:


# find the age with decimal

df_titanic[df_titanic['Age'].str.find(',') > -1]


# In[135]:


# change the comma into period, and minus values to positive values

df_titanic['Age'] = df_titanic['Age'].str.replace(',','.')
#df_titanic['Age'] = df_titanic['Age'].str.replace('-3','3')
#df_titanic['Age'] = df_titanic['Age'].str.replace('-12','12')
#df_titanic['Age'] = df_titanic['Age'].str.replace('4435',np.NaN)
df_titanic.tail()


# In[136]:


#Change the data type 

#Age
df_titanic['Age'] = df_titanic['Age'].astype(float)

df_titanic.info()


# In[137]:


# round the numbers of age

df_titanic['Age'] = np.round(df_titanic['Age'])
df_titanic.tail(5)


# ### Pclass

# In[138]:


df_titanic['Pclass'].value_counts()


# There is a value that does not match with other values, probably it is a typo. Thus, we need to change it to proper value.

# In[139]:


df_titanic['Pclass'] = df_titanic['Pclass'].replace(-2,2)
df_titanic['Pclass'].value_counts()


# In[140]:


#change the data type

df_titanic['Pclass'] = df_titanic['Pclass'].astype(int)

df_titanic.info()


# ### Parch

# In[141]:


df_titanic['Parch'].value_counts()


# In[142]:


df_titanic[df_titanic['Parch'].str.contains('no')]


# There is a inconsistent value in Parch column, so we need to change it. this time we will change it to the NaN value and it will be executed later when handling missing values.

# In[143]:


#changing the value "no" with mode 

df_titanic['Parch'] = df_titanic['Parch'].replace('no', np.NaN)

df_titanic['Parch'].value_counts()


# In[144]:


#change the data type

df_titanic['Parch'] = df_titanic['Parch'].astype(float)
df_titanic.info()


# ### Fare

# In[145]:


df_titanic[df_titanic['Fare'].str.find('a') > -1]


# We will change the inconsitent value with NaN value.

# In[146]:


df_titanic['Fare'] = df_titanic['Fare'].str.replace(',','.')
df_titanic['Fare'] = df_titanic['Fare'].replace('07.maj', np.NaN)


# In[147]:


#change the data type and round the numbers

df_titanic['Fare'] = df_titanic['Fare'].astype(float)
df_titanic['Fare'] = np.round(df_titanic['Fare'],2)
df_titanic.tail(5)


# ### Embarked

# In[148]:


df_titanic['Embarked'].value_counts()


# We will change the value "So","Co","Qe" to "S","Q","C".

# In[149]:


#change the label

# "So"
df_titanic['Embarked'] = df_titanic['Embarked'].replace('So','S')

# "Co"
df_titanic['Embarked'] = df_titanic['Embarked'].replace('Co','C')

# "Qe"
df_titanic['Embarked'] = df_titanic['Embarked'].replace('Qe','Q')

df_titanic['Embarked'].value_counts()


# ### Ship
# 

# In[150]:


df_titanic['ship'].value_counts()


# In[151]:


#Change the inconsistent values 

df_titanic['ship'] = df_titanic['ship'].replace(['Titani','Titnic'],'Titanic')
df_titanic['ship'].value_counts()


# In[152]:


#Change the column name into Title
df_titanic.rename(columns = {'ship':'Ship'}, inplace = True)
df_titanic.head(1)


# ## Handling missing values

# In[153]:


df_titanic.isna().sum()


# In[154]:


df_titanic.isnull().mean()


# We can see that age, ticket, fare, cabin and embarked columns have missing values. 

# ### Age column

# we will fill the missing values with median value of age. 

# In[155]:


sns.distplot(df_titanic.Age)
plt.show()


# In[156]:


df_titanic['Age'].median()


# In[157]:


df_titanic['Age'] = df_titanic['Age'].fillna(df_titanic['Age'].median())
df_titanic.isna().sum()

#df['rating'] = df['rating'].fillna(df['rating'].median())


# ### Parch column

# we will fill the missing values with median values.

# In[158]:


df_titanic['Parch'].value_counts()


# In[159]:


df_titanic['Parch'].isna().sum()


# In[160]:


df_titanic['Parch'] = df_titanic['Parch'].fillna(df_titanic['Parch'].median())
df_titanic['Parch'].isna().sum()


# ### Ticket column
# 

# In[161]:


df_titanic['Ticket'].value_counts()


# we will fill the missing value in ticket column with most frequent value.

# In[162]:


df_titanic['Ticket'].mode()


# In[163]:


df_titanic['Ticket'].fillna(df_titanic['Ticket'].mode()[0], inplace = True)
df_titanic.isna().sum()


# In[164]:


df_titanic['Fare'].value_counts()


# ### Fare Column

# we will fill the fare column with median value because it looks like the data is right skewed. Thus, it is better to use median.

# In[165]:


sns.distplot(df_titanic.Fare)
plt.show()


# In[166]:


df_titanic['Fare'] = df_titanic['Fare'].fillna(df_titanic['Fare'].median())
df_titanic.isna().sum()


# ### Cabin Column

# In[167]:


df_titanic['Cabin'].mode()


# Since it is categorical value, we will change the missing values with the most frequent values.

# In[168]:


df_titanic['Cabin'].fillna(df_titanic['Cabin'].mode()[0], inplace = True)
df_titanic.isna().sum()


# ### Embarked Column

# In[169]:


df_titanic['Embarked'].mode()


# we will change the missing values with the most frequent values.

# In[170]:


df_titanic['Embarked'].fillna(df_titanic['Embarked'].mode()[0], inplace = True)
df_titanic.isna().sum()


# ## Handling outlier

# In[171]:


df_titanic.describe()


# We will handle the outlier of those six columns above.

# ### Survived

# In[172]:


df_titanic['Survived'].describe()


# In[173]:


df_titanic["Survived"].value_counts()


# In[174]:


df_titanic.boxplot(column=['Survived'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)
plt.show()


# There is no outlier in the survived column.

# ### Pclass

# In[175]:


df_titanic['Pclass'].describe()


# In[176]:


df_titanic.boxplot(column=['Pclass'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)
plt.show()


# There is no outlier in Pclass column.

# ### Age

# In[177]:


df_titanic['Age'].describe()


# In[178]:


df_titanic['Age'].value_counts()


# In[179]:


sns.distplot(df_titanic.Age)
plt.show()


# In[180]:


df_titanic.boxplot(column=['Age'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)
plt.show()


# In[181]:


# quartile 1 and 3
Q1 = df_titanic['Age'].quantile(0.25)
Q3 = df_titanic['Age'].quantile(0.75)

#IQR
IQR = Q3 - Q1
min_age = Q1 - 1.5 * IQR
max_age = Q3 + 1.5 * IQR

print('Q1:\n',Q1)
print('\nQ3:\n',Q3)
print('\nIQR:\n',IQR)
print('\nMin:\n',min_age)
print('\nMax:\n',max_age)


# In[182]:


#another way to detect outliers

outliers = []
def detect_outliers_iqr(data):
    data = sorted(df_titanic['Age'])
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers  #Driver code
sample_outliers = detect_outliers_iqr(df_titanic['Age'])
print("Outliers from IQR method: ", sample_outliers)


# As we can see there are some outliers, to handle the outliers of this column, we will replace the outliers with median values.

# In[183]:


median = float(df_titanic['Age'].median())

df_titanic['Age'] = np.where(df_titanic['Age'] <2.5, median, df_titanic['Age'])
df_titanic['Age'] = np.where(df_titanic['Age'] >54.5, median, df_titanic['Age'])
df_titanic.boxplot(column=['Age'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)


# In[184]:


sns.distplot(df_titanic.Age)


# ### SibSp

# In[185]:


df_titanic['SibSp'].describe()


# In[186]:


df_titanic['SibSp'].value_counts()


# In[187]:


df_titanic.boxplot(column=['SibSp'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)


# In[188]:


sns.distplot(df_titanic.SibSp)


# It looks like the data is skewed, thus we are going to handle the outliers with median as well.

# In[189]:


# quartile 1 and 3
Q1 = df_titanic['SibSp'].quantile(0.25)
Q3 = df_titanic['SibSp'].quantile(0.75)

#IQR
IQR = Q3 - Q1
min_sibsp = Q1 - 1.5 * IQR
max_sibsp = Q3 + 1.5 * IQR

print('Q1:\n',Q1)
print('\nQ3:\n',Q3)
print('\nIQR:\n',IQR)
print('\nMin:\n',min_sibsp)
print('\nMax:\n',max_sibsp)


# In[190]:


median_sibsp = float(df_titanic['SibSp'].median())

df_titanic['SibSp'] = np.where(df_titanic['SibSp'] <-1.5, median_sibsp, df_titanic['SibSp'])
df_titanic['SibSp'] = np.where(df_titanic['SibSp'] >2.5, median_sibsp, df_titanic['SibSp'])
df_titanic.boxplot(column=['SibSp'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)


# In[191]:


df_titanic['SibSp'].value_counts()


# ### Parch

# In[192]:


df_titanic['Parch'].describe()


# In[193]:


df_titanic['Parch'].value_counts()


# In[194]:


df_titanic.boxplot(column=['Parch'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)


# In[195]:


sns.distplot(df_titanic.Parch)


# In[196]:


# quartile 1 and 3
Q1 = df_titanic['Parch'].quantile(0.25)
Q3 = df_titanic['Parch'].quantile(0.75)

#IQR
IQR = Q3 - Q1
min_parch = Q1 - 1.5 * IQR
max_parch = Q3 + 1.5 * IQR

print('Q1:\n',Q1)
print('\nQ3:\n',Q3)
print('\nIQR:\n',IQR)
print('\nMin:\n',min_parch)
print('\nMax:\n',max_parch)


# In[197]:


df_titanic['Parch'] = np.where(df_titanic['Parch'] >0, median_sibsp, df_titanic['Parch'])
df_titanic.boxplot(column=['Parch'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)


# In[198]:


sns.distplot(df_titanic.Parch)


# ### Fare

# In[199]:


df_titanic.boxplot(column=['Fare'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)


# In[200]:


sns.distplot(df_titanic.Fare)
plt.show()


# In[201]:


# quartile 1 and 3
Q1 = df_titanic['Fare'].quantile(0.25)
Q3 = df_titanic['Fare'].quantile(0.75)

#IQR
IQR = Q3 - Q1
min_fare = Q1 - 1.5 * IQR
max_fare = Q3 + 1.5 * IQR

print('Q1:\n',Q1)
print('\nQ3:\n',Q3)
print('\nIQR:\n',IQR)
print('\nMin:\n',min_fare)
print('\nMax:\n',max_fare)


# In[202]:


#another way to detect outliers

outliers = []
def detect_outliers_iqr(data):
    data = sorted(df_titanic['Fare'])
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers  #Driver code
sample_outliers = detect_outliers_iqr(df_titanic['Fare'])
print("Outliers from IQR method: ", sample_outliers)


# There are some outliers, we also will handle this column with median values.

# In[203]:


median_fare = float(df_titanic['Fare'].median())

df_titanic['Fare'] = np.where(df_titanic['Fare'] <-26, median_fare, df_titanic['Fare'])
df_titanic['Fare'] = np.where(df_titanic['Fare'] >64.8, median_fare, df_titanic['Fare'])
df_titanic.boxplot(column=['Fare'],fontsize=10,rot=0,grid=False,figsize=(5,5),vert=False)


# In[204]:


sns.distplot(df_titanic.Fare)


# #### do one more checking for the null values

# In[205]:


df_titanic.isna().sum()


# ### Normalization

# In[206]:


df_titanic.describe()


# In[207]:


df_scaled = df_titanic[["Survived","Pclass","Age","SibSp","Parch","Fare"]]


# In[208]:


# apply normalization techniques
for column in df_scaled.columns:
    df_scaled[column] = (df_scaled[column] - df_scaled[column].min()) / (df_scaled[column].max() - df_scaled[column].min())    
  
# view normalized data
df_scaled


# In[209]:


df_scaled.hist(bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
plt.show()


# In[210]:


df_titanic.info()


# In[211]:


df_titanic.describe()


# In[212]:


# saving the dataframe
df_titanic.to_csv('titanic_clean.csv')


# In[ ]:




