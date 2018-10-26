import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import random_projection

df_train = np.loadtxt('./dataset/train.dat')
X_train = pd.DataFrame(df_train)
df_test = np.loadtxt('./dataset/test.dat')
X_test = pd.DataFrame(df_test)
df_train_labels = np.loadtxt('./dataset/train.labels')
y_train = pd.DataFrame(df_train_labels)

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
svd = TruncatedSVD(n_components=30)
svd.fit(X_train)

X_train_svd = svd.transform(X_train)
X_test_svd = svd.transform(X_test)
X_train_svd_train, X_train_svd_test, y_train_train, y_train_test = train_test_split(
    X_train_svd, y_train, test_size=0.1)
y_train_train = np.ravel(y_train_train)
knn_classifier.fit(X_train_svd_train, y_train_train)
y_pred = knn_classifier.predict(X_train_svd_test)

print(metrics.classification_report(y_train_test, y_pred))

knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
rp = random_projection.SparseRandomProjection(n_components=30)
rp.fit(X_train)

X_train_rp = rp.transform(X_train)
X_test_rp = rp.transform(X_test)
X_train_rp_train, X_train_rp_test, y_train_train, y_train_test = train_test_split(
    X_train_rp, y_train, test_size=0.1)
y_train_train = np.ravel(y_train_train)
knn_classifier.fit(X_train_rp_train, y_train_train)
y_pred = knn_classifier.predict(X_train_rp_test)

print(metrics.classification_report(y_train_test, y_pred))

pca = PCA(n_components=30)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
X_train_pca_train, X_train_pca_test, y_train_train, y_train_test = train_test_split(
    X_train_pca, y_train, test_size=0.1)

y_train_train = np.ravel(y_train_train)
knn_classifier.fit(X_train_pca_train, y_train_train)
y_pred = knn_classifier.predict(X_train_pca_test)
print(metrics.classification_report(y_train_test, y_pred))

knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
svd = TruncatedSVD(n_components=30)
svd.fit(X_train)

X_train_svd = svd.transform(X_train)
X_test_svd = svd.transform(X_test)

y_train = np.ravel(y_train)
knn_classifier.fit(X_train_svd, y_train)
y_svd_pred = knn_classifier.predict(X_test_svd)


knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
rp = random_projection.SparseRandomProjection(n_components=30)
rp.fit(X_train)

X_train_rp = rp.transform(X_train)
X_test_rp = rp.transform(X_test)

y_train = np.ravel(y_train)
knn_classifier.fit(X_train_rp, y_train)
y_rp_pred = knn_classifier.predict(X_test_rp)


knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
pca = PCA(n_components=30)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn_classifier.fit(X_train_pca, y_train)
y_pca_pred = knn_classifier.predict(X_test_pca)

df_pred = []
for i in range(0, 5296):
    if y_svd_pred[i] == y_rp_pred[i]:
        df_pred.append(y_rp_pred[i])
    elif y_svd_pred[i] == y_pca_pred[i]:
        df_pred.append(y_svd_pred[i])
    elif y_rp_pred[i] == y_pca_pred[i]:
        df_pred.append(y_rp_pred[i])
    elif (y_svd_pred[i] - y_rp_pred[i]) < (y_svd_pred[i] - y_pca_pred[i]):
        df_pred.append(y_rp_pred[i])
    else:
        df_pred.append(y_pca_pred[i])

df_pred = pd.DataFrame(df_pred)

predicted_int = df_pred.astype('int', copy=False)

np.savetxt('./predictions/predictions.txt',
           predicted_int, fmt='%s', delimiter="\n")
