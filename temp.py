import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

X = np.array([1,0,1,1,1,1,0,1])
Y = np.array([1,0,1,1,1,1,0,1])
print X.shape
X = X.reshape(-1, 1)
# Y = Y.reshape(-1, 1)
print X.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print clf.score(X_test, y_test)
