
# coding: utf-8

# In[271]:


import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from matplotlib import pyplot as plt
from sklearn import linear_model,decomposition
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.feature_selection import RFE
from sklearn import svm
from keras.models import Sequential 
from keras.optimizers import SGD
from keras.layers import Dense , Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import pickle
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df1 = pd.read_csv('Final_Development set_With corrected dates and errors_Dimension1.csv',delimiter=';')
df2 = pd.read_csv('Final_Development set_With corrected dates and errors_Dimension2.csv',delimiter=';')
#remove duplicate rows in both dataframe caused during data extraction
df2 = df2.drop_duplicates()
df1 = df1.drop_duplicates()


# In[3]:


df1.info()


# In[4]:


df2.info()


# In[5]:


df1.columns = [w.replace('.','_') for w in df1.columns]
df2.columns = [w.replace('.','_') for w in df2.columns]


# In[6]:


df1.head()


# In[7]:


df2.head()


# In[8]:


df2['Success'] = np.where(df2['Sucess'] == True,1,0)
df1['Success'] = np.where(df1['Success']==True,1,0)
del df2['Sucess']


# In[9]:


df1.head()


# In[10]:


df2.head()


# In[11]:


df1.sort_values('ga_dateHourMinute').head(n=10)


# In[12]:


df2.sort_values('ga_dateHourMinute').head(n=10)


# In[13]:


df2['ga_uniqueDimensionCombinations'].unique()


# In[14]:


mergedf = pd.merge(df1,df2,how = 'inner',on = ["ga_dateHourMinute","ga_sessionDurationBucket","ga_sessionsWithEvent","Success"])


# In[15]:


mergedf.shape


# In[16]:


mergedf.sort_values('ga_dateHourMinute').head()


# In[17]:


mergedf.info()


# In[18]:


group = mergedf.groupby(["ga_dateHourMinute","ga_sessionDurationBucket","ga_sessionsWithEvent","Success","ga_userBucket" ,"Unique_code"]).size()


# In[19]:


#no dups
group[group > 1]


# In[20]:


mergedf.head()


# In[21]:


# data cleaning and transformation


# In[22]:


print("Unique browsers = {}".format(mergedf['ga_browser_encoded'].nunique()))
print("Unique OS = {}".format(mergedf['ga_operatingSystem_encoded'].nunique()))
print("Unique OSV = {}".format(mergedf['ga_operatingSystemVersion_encoded'].nunique()))
print("Unique Language model = {}".format(mergedf['ga_language_encoded'].nunique()))
print("Unique Device = {}".format(mergedf['ga_DeviceInfo_encoded'].nunique()))
print("Unique Codes  = {}".format(mergedf['Unique_code'].nunique()))


# In[23]:


# its better to hot encode individual features than the combined one

#ohe_codes = preprocessing.OneHotEncoder(categorical_features=['ga_browser_encoded','ga_operatingSystem_encoded','ga_operatingSystemVersion_encoded','ga_language_encoded','ga_DeviceInfo_encoded'], dtype=np.object)
data=  mergedf.loc[:,['ga_browser_encoded','ga_operatingSystem_encoded','ga_operatingSystemVersion_encoded','ga_language_encoded','ga_DeviceInfo_encoded']]
ohe_codes = pd.get_dummies(data)


# In[24]:


ohe_codes.shape


# In[25]:


mergedf.drop(['ga_browser_encoded','ga_operatingSystem_encoded','ga_operatingSystemVersion_encoded','ga_language_encoded','ga_DeviceInfo_encoded'],inplace=True, axis=1)
mergedf = mergedf.join(ohe_codes)


# In[26]:


mergedf.info()


# In[27]:


mergedf.head()


# In[28]:


#drop the unqiue code col for now
unqcde = mergedf['Unique_code']
del mergedf['Unique_code']


# In[29]:


mergedf.shape


# In[30]:


mergedf['ga_userType'].nunique()


# In[31]:


mergedf['ga_userType'] = np.where(mergedf['ga_userType'] == 'New Visitor',1,0)


# In[32]:


mergedf.iloc[:,:9].head()


# In[33]:


mergedf.info()
#all features are now numeric


# In[34]:


# some visualizations


# In[35]:


#classes are not much imbalanced
sns.countplot(x = mergedf['Success'])


# In[36]:


b  = sns.pairplot(mergedf.iloc[:,:8])


# In[37]:


sns.set_style("dark")
b  = sns.pairplot(mergedf.iloc[:,:8],x_vars=["ga_dateHourMinute"],y_vars=["ga_sessionCount"])
b.fig.set_size_inches(12,6)
#in below plot, data ranges from march to june inclusive and session count of each month is in increaasing order


# In[38]:


sns.barplot(x=mergedf['Success'],y=mergedf['ga_sessionCount'])
#avg session count of succsessful txn is lesser 


# In[39]:


b  = sns.pairplot(mergedf.iloc[:,:8],x_vars=["ga_sessionDurationBucket"],y_vars=["ga_sessionCount"],hue="Success")
b.fig.set_size_inches(12,6)
#those txn with less session duration bucket and less session count are more successful


# In[40]:


b  = sns.pairplot(mergedf.iloc[:,:8],x_vars=["ga_userBucket"],y_vars=["ga_sessionCount"],hue="Success")
b.fig.set_size_inches(12,6)
#


# In[41]:


g = sns.countplot(x=unqcde)
#one particular unique code repeat alot


# In[42]:


unqcde.describe()
#A4B8C2D77E359 -> this code occurs 2736 times as the most common combination of mobile, os, os version etc


# In[43]:


null_data = mergedf[mergedf.isnull().any(axis=1)]
print("Number of rows with missing data = {}".format(len(null_data)))


# In[44]:


# Data normalization / scaling 
y = mergedf['Success']
X = mergedf.loc[:,mergedf.columns!='Success']
minmiax  = preprocessing.MinMaxScaler()
merge_scale = pd.DataFrame(minmiax.fit_transform(X.values))
#merge_norm = preprocessing.MinMaxScaler.fit_transform(X,y)
#merge_standard = preprocessing.scale(mergedf)


# In[45]:


merge_scale.columns = X.columns
merge_scale.head()


# In[46]:


merge_standard = pd.DataFrame(preprocessing.scale(X.values))


# In[47]:


merge_standard.columns = X.columns
merge_standard.head()


# In[48]:


#split data into train/test set


# In[49]:


X_train_norm, X_test_norm, y_train_norm, y_test_norm = model_selection.train_test_split(merge_scale,y,test_size = 0.33,random_state = 42)
#X_train_scale, X_test_scale, y_train_scale, y_test_scale = model_selection.train_test_split(merge_scale,y,test_size = 0.33,random_state = 42)



# In[235]:


## split for neural network - 90/10 split
X_train_nn, X_test_nn, y_train_nn, y_test_nn = model_selection.train_test_split(merge_scale,y,test_size = 0.10,random_state = 42)


# In[50]:


# modelling 


# In[51]:


logit = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca',pca),('logistic',logit)])


# In[52]:


pca.fit(X_train_norm)


# In[53]:


plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
# Prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = model_selection.GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_train_norm, y_train_norm)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()


# In[54]:


estimator.best_estimator_.named_steps['pca'].n_components


# In[55]:


estimator.best_score_


# In[56]:


y_pred = estimator.best_estimator_.predict(X_test_norm)


# In[57]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test_norm, y_pred)
print(confusion_matrix)


# In[61]:


logreg = linear_model.LogisticRegression()
rfe = RFE(logreg , 40)
rfe = rfe.fit(X_train_norm,y_train_norm)


# In[62]:


print(rfe.support_)
print(rfe.ranking_)


# In[72]:


cols = [X_train_norm.columns[idx] for idx,i in enumerate(rfe.ranking_) if rfe.support_[idx] ]


# In[74]:


#selected columns
cols


# In[87]:


X_train_final = X_train_norm[cols]
y_train_final = y_train_norm
X_test_final = X_test_norm[cols]
y_test_final = y_test_norm


# In[83]:


y_final.values


# In[95]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(max_iter=500,verbose=1)
logreg.fit(X_train_norm, y_train_norm)


# In[96]:


y_pred = logreg.predict(X_test_norm)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_norm, y_test_norm)))


# In[97]:


#CV
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=20, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train_norm, y_train_norm, cv=kfold, scoring=scoring)

print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# In[102]:


print(confusionmatrix(y_test_norm,y_pred))


# In[103]:


## random forest


# In[109]:


rf1 = RandomForestClassifier(n_estimators=100)
rf1.fit(X_train_norm,y_train_norm)
y_pred_rf1=rf1.predict(X_test_norm)
print(confusionmatrix(y_test_norm,y_pred_rf1))
print("Accuracy:",metrics.accuracy_score(y_test_norm, y_pred_rf1))


# In[138]:


rf2 = RandomForestClassifier(n_estimators=200,random_state=42)
rf2.fit(X_train_norm,y_train_norm)
y_pred_rf2=rf2.predict(X_test_norm)
print(confusionmatrix(y_test_norm,y_pred_rf2))
print("Accuracy:",metrics.accuracy_score(y_test_norm, y_pred_rf2))


# In[139]:


rf3 = RandomForestClassifier(n_estimators=200,random_state=42)
rf3.fit(X_train_final,y_train_final)
y_pred_rf3=rf3.predict(X_test_final)
print(confusionmatrix(y_test_final,y_pred_rf3))
print("Accuracy:",metrics.accuracy_score(y_test_final, y_pred_rf3))


# In[143]:


rf4 = RandomForestClassifier(n_estimators=200,max_features="log2",random_state=42,oob_score=True)
rf4.fit(X_train_norm,y_train_norm)
y_pred_rf4=rf4.predict(X_test_norm)
print(confusionmatrix(y_test_norm,y_pred_rf4))
print("Accuracy:",metrics.accuracy_score(y_test_norm, y_pred_rf4))


# In[145]:


rf4.oob_score_


# In[132]:


feature_imp = pd.Series(rf2.feature_importances_,index=X_train_norm.columns).sort_values(ascending=False)
#feature_imp

sns.barplot(x=feature_imp[:20], y=feature_imp.index[:20])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[170]:


#Metric


# In[168]:


False_Pos_Rate,True_Pos_Rate,ROC_CURVE = roccurve(y_pred_rf2,y_test_norm)
plotroc(False_Pos_Rate,True_Pos_Rate,ROC_CURVE,'Receiver operating characteristic for Model - RF2')


# In[169]:


False_Pos_Rate,True_Pos_Rate,ROC_CURVE = roccurve(y_pred_rf3,y_test_norm)
plotroc(False_Pos_Rate,True_Pos_Rate,ROC_CURVE,'Receiver operating characteristic for Model - RF3')


# In[173]:


#SVM 


# In[175]:


svm1 = svm.SVC(kernel='linear') # Linear Kernel
svm1.fit(X_train_norm,y_train_norm)
y_pred_svm1 = svm1.predict(X_test_norm)
print(confusionmatrix(y_test_norm,y_pred_svm1))
print("Accuracy:",metrics.accuracy_score(y_test_norm, y_pred_svm1))


# In[176]:






# In[177]:



# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=1.0,random_state=42).fit(X_train_norm, y_train_norm)

y_pred_svm2 = lin_svc.predict(X_test_norm)
print(confusionmatrix(y_test_norm,y_pred_svm2))
print("Accuracy:",metrics.accuracy_score(y_test_norm, y_pred_svm2))


# In[178]:


# SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0,random_state=42).fit(X_train_norm, y_train_norm)


y_pred_svm3 = rbf_svc.predict(X_test_norm)
print(confusionmatrix(y_test_norm,y_pred_svm3))
print("Accuracy:",metrics.accuracy_score(y_test_norm, y_pred_svm3))


# In[179]:


# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, C=1.0,random_state=42).fit(X_train_norm, y_train_norm)
y_pred_svm4 = poly_svc.predict(X_test_norm)
print(confusionmatrix(y_test_norm,y_pred_svm4))
print("Accuracy:",metrics.accuracy_score(y_test_norm, y_pred_svm4))


# In[188]:




y_pred_svm5 = rbf_svc1.predict(X_test_norm)
print(confusionmatrix(y_test_norm,y_pred_svm5))
print("Accuracy:",metrics.accuracy_score(y_test_norm, y_pred_svm5))


# 0.71 -  (c 1, g 0.8)
# 0.68  - (c 0.1 g 0.8  )
# 0.7240467717336044 - (c 10, g-1)
# 0.710930350788002 - c 10, g 10


# In[187]:


# SVC with RBF kernel
rbf_svc1 = svm.SVC(kernel='rbf', gamma=10, C=10,random_state=42).fit(X_train_norm, y_train_norm)


# In[243]:


## neural network
epochs = 500
batch_size = 128
dropout_rate = 0.4
model1 = model_NN(X_train_nn.shape[1],dropout_rate)
print(model1.summary())
model1 = train_NN(model1,X_train_nn,y_train_nn,epochs,batch_size)
eval_NN(model1,X_test_nn,y_test_nn)


# In[241]:


plotgraphs(model1.model.history)


# In[249]:


#Total params: 251,401
#cv score - 83.26%
plotgraphs(model1.model.history)
y_pred_nn = model1.predict(X_test_nn)
False_Pos_Rate,True_Pos_Rate,ROC_CURVE = roccurve(y_pred_nn,y_test_nn)
plotroc(False_Pos_Rate,True_Pos_Rate,ROC_CURVE,'Receiver operating characteristic for Model - NN')
yaml_filename = 'model1_500.yaml'
hd5_filename= 'model1_500.hd5'
savemodel(model1,yaml_filename,hd5_filename)


# In[262]:


X_train_nn.shape


# In[269]:


## neural network
epochs = 400
batch_size = 256
learning_rate = 0.1
momentum  = 0.7
decay_rate = learning_rate / epochs
model2 = model_NN2(X_train_nn.shape[1],learning_rate,momentum,decay_rate)
print(model2.summary())
model2 = train_NN(model2,X_train_nn,y_train_nn,epochs,batch_size)
eval_NN(model2,X_test_nn,y_test_nn)


# In[270]:


#post training ...
plotgraphs(model2.model.history)
y_pred_nn = model2.predict(X_test_nn)
False_Pos_Rate,True_Pos_Rate,ROC_CURVE = roccurve(y_pred_nn,y_test_nn)
plotroc(False_Pos_Rate,True_Pos_Rate,ROC_CURVE,'Receiver operating characteristic for Model - SGD NN')
yaml_filename = 'model2_500.yaml'
hd5_filename= 'model2_500.hd5'
#savemodel(model1,yaml_filename,hd5_filename)


# In[ ]:



model=xgb.XGBClassifier(random_state=42,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# ## function definitions

# In[167]:


def roccurve(y_score,y_test):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area
    return fpr,tpr,roc_auc

def plotroc(fpr,tpr,roc_auc,model_text):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_text)
    plt.legend(loc="lower right")
    plt.show()


# In[171]:


def confusionmatrix(y_test,y_pred):
    return classification_report(y_test, y_pred)


# In[268]:


def model_NN(input_shape,dropout_rate):
    
    # create model
    model = Sequential()
    model.add(Dense(300, input_dim=input_shape, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(50, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def model_NN2(input_shape,learn_rate,momentum,decay_rate):
    
    # create model with stochastic gradient descent optimizer
    model = Sequential()
    model.add(Dense(300, input_dim=input_shape, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(50, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=learn_rate, momentum=momentum,decay=decay_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
    
def train_NN(model,X_train,y_train,epochs,batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.25,verbose=0)
    return model

def eval_NN(model,X_test,y_test):
    # evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def plotgraphs(model):
    # list all data in history
#    print(model.model.keys())
    # summarize history for accuracy
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def savemodel(model,yaml_filename,hd5_filename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(yaml_filename, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(hd5_filename)
    print("Saved model to disk")

