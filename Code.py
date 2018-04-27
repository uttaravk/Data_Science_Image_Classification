
# coding: utf-8

# In[195]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import random_projection


# In[196]:


d1=np.loadtxt('/Users/uttara/desktop/data/train.dat')
X_train=pd.DataFrame(d1)
d2=np.loadtxt('/Users/uttara/desktop/data/test.dat')
X_test=pd.DataFrame(d2)
d3=np.loadtxt('/Users/uttara/desktop/data/train.labels')
y_train=pd.DataFrame(d3)


# In[197]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
svd = TruncatedSVD(n_components=30)
svd.fit(X_train)

X_train_svd1 = svd.transform(X_train)
X_test_svd1 = svd.transform(X_test)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_svd1, y_train, test_size=0.1)
clf.fit(X_train1, y_train1)
y_pred=clf.predict(X_test1)

print(metrics.classification_report(y_test1, y_pred))


# In[198]:


clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
rp = random_projection.SparseRandomProjection(n_components=80)
rp.fit(X_train)

X_train_rp1 = rp.transform(X_train)
X_test_rp1 = rp.transform(X_test)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_rp1, y_train, test_size=0.1)
clf.fit(X_train1, y_train1)
y_pred=clf.predict(X_test1)

print(metrics.classification_report(y_test1, y_pred))


# In[199]:


pca = PCA(n_components=150)
pca.fit(X_train)

X_train_pca1 = pca.transform(X_train)
X_test_pca1 = pca.transform(X_test)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_svd1, y_train, test_size=0.1)
clf.fit(X_train1, y_train1)
y_pred=clf.predict(X_test1)
print(metrics.classification_report(y_test1, y_pred))


# In[200]:


clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
svd = TruncatedSVD(n_components=30)
svd.fit(X_train)

X_train_svd = svd.transform(X_train)
X_test_svd = svd.transform(X_test)

clf.fit(X_train_svd, y_train)
y1 = clf.predict(X_test_svd)


# In[201]:


clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
rp = random_projection.SparseRandomProjection(n_components=80)
rp.fit(X_train)

X_train_rp = rp.transform(X_train)
X_test_rp = rp.transform(X_test)

clf.fit(X_train_rp, y_train)
y2 = clf.predict(X_test_rp)


# In[202]:


clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
pca = PCA(n_components=150)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

clf.fit(X_train_pca, y_train)
y3 = clf.predict(X_test_pca)


# In[203]:


predicted_df=[]
for i in range (0,5296):
    if y1[i]==y2[i]:
        predicted_df.append(y2[i])
    elif y1[i]==y3[i]:
        predicted_df.append(y1[i])
    elif y2[i]==y3[1]:
        predicted_df.append(y2[i])
    elif (y1[i]-y2[i])<(y1[i]-y3[i]):
        predicted_df.append(y2[i])
    else:
        predicted_df.append(y3[i])

predicted_df=pd.DataFrame(predicted_df)


# In[204]:


predicted_int = predicted_df.astype('int', copy=False)


# In[205]:


np.savetxt('/Users/uttara/desktop/predicted_f1.0.txt',predicted_int, fmt='%s', delimiter="\n" )

