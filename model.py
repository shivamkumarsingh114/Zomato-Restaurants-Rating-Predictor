
# coding: utf-8

# In[20]:


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
from sklearn.model_selection import cross_validate
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
import pickle


# In[21]:


#Reading the Dataset
df1 = pd.read_csv('zomato_res_final.csv' , delimiter=',')
test_ds = pd.read_csv('test.csv' , delimiter=',')

#Data Pre-processing
#Since we scraped the data it is quite clean just removed the reastaurants with no rating(i.e New restaturants)
#Also removed restaurants with cost of two as zero
df1 = df1[df1.Rating != 0]
df1 = df1[df1.Cost != 0]
test_ds = test_ds[test_ds.Rating != 0]
test_ds = test_ds[test_ds.Cost != 0]
df1.head()


# In[22]:


#Conversion of data types to float and int
def preprocessing(df1):
    df1['Rating'] = pd.to_numeric(df1['Rating'], errors='coerce')
    df1['Featured_in'] = pd.to_numeric(df1['Featured_in'], errors='coerce')
    df1['Votes'] = pd.to_numeric(df1['Votes'], errors='coerce')
    df1['Type'] = pd.to_numeric(df1['Type'], errors='coerce')
    df1['Outlets'] = pd.to_numeric(df1['Outlets'], errors='coerce')
    df1['Cost'] = pd.to_numeric(df1['Cost'], errors='coerce')
    return df1
df1 = preprocessing(df1)
test_ds = preprocessing(test_ds)
df1.dtypes


# In[23]:


#Features that will be used for model training
features = ["Cost" , "Cuisine" ,"Featured_in", "Votes" , "Type" , "Book_table" , "Outlets" , "Delivery" , "Call" , "View_menu"]


# In[24]:


params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 0,
          "seed": 42
          }


# In[25]:


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(250, 0, as_cmap=True),
                square=True, ax=ax)
plot_corr(df1)



# In[26]:


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}
plt.rc('font', **font)
f, ax = plt.subplots(1,1, figsize = (15, 4))
ax = sns.countplot(df1[df1['Rating'] != 0]['Rating'])
plt.show()

# In[27]:


plt.rcParams['figure.figsize']=(6,6)
plt.scatter(df1['Rating'], df1['Cost'])
plt.title('Scatter plot between Cost and Ratings')
plt.xlabel('Rating')
plt.ylabel('Cost')
plt.show()
plt.scatter(df1['Rating'], df1['Votes'])
plt.title('Scatter plot between Votes and ratings')
plt.xlabel('Rating')
plt.ylabel('Votes')
plt.show()
plt.scatter(df1['Rating'], df1['Featured_in'])
plt.title('Scatter plot between Featured_in and Ratings')
plt.xlabel('Rating')
plt.ylabel('Featured_in')
plt.show()
plt.scatter(df1['Rating'], df1['Outlets'])
plt.title('Scatter plot between Extra Outlets and Ratings')
plt.xlabel('Rating')
plt.ylabel('Total extra outlets')


# In[28]:


#Replaced Nan values with 0
df1.fillna(0, inplace=True)
train, test = train_test_split(df1, test_size=0.3 , random_state = 5)
test_label = test["Rating"]


# In[29]:


#Evaluation metrics for xgboost
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe
def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

num_trees = 100
print("Train a XGBoost model")
val_size = 100000
X_train, X_test = train_test_split(df1, test_size=0.01)
dtrain = xgb.DMatrix(X_train[features], X_train["Rating"])
dvalid = xgb.DMatrix(X_test[features], X_test["Rating"])
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)
filename="submission.sav"
pickle.dump(gbm, open(filename,"wb"))

# In[30]:


fig, ax = plt.subplots(figsize=(6,12))
xgb.plot_importance(gbm, max_num_features=50, height=0.8, ax=ax)
plt.show()


# In[31]:


print("Validating")
train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
error = rmspe(train_probs, X_test['Rating'].values)
print('Accuracy', (1-error)*100)

print("Predicting")
train_probs = gbm.predict(xgb.DMatrix(test_ds[features]))
error = rmspe(train_probs, test_ds['Rating'].values)
print ('Accuracy' , (1 - error)*100)
test_probs = gbm.predict(xgb.DMatrix(test_ds[features]))
submission = pd.DataFrame({"Name": test_ds["Name"], "Rating": test_probs})
submission.to_csv("test_submission.csv", index=False)

# In[32]:


l = (df1["Location"]).value_counts()
l = l [0:25]
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 60}
plt.rc('font', **font)
a=l.plot(figsize = (60,30) , kind = 'bar',label="No. of restaurants vs location")
a.set_ylabel("No. of restaurants")


# In[ ]:
