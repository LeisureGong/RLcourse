import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
print('make classification ...')
X,y = make_classification(n_samples=1000000,
                         n_features=50,
                         n_informative=30,
                         n_redundant=5,
                         n_repeated=0,
                         n_classes=2,
                         n_clusters_per_class=2,
                         class_sep=1,
                         flip_y=0.01,
                         weights=[0.5,0.5],
                         random_state=17)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1000)
print(f'X_train shape: {X_train.shape}')
print(f'Train LGBM classifier ...')
clf = LGBMClassifier(n_estimators=100,
                     num_leaves=64,
                     max_depth=5,
                     learning_rate=0.1,
                     random_state=1000,
                     n_jobs=-1)
start = time.time()
clf.fit(X_train,y_train)
elapsed = time.time() - start
print(f'LGBM Training ran in {elapsed:.5f} seconds')
y_pred = clf.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test,y_pred):.2f}')
print(f'Train XGB classifier ...')
clf = XGBClassifier(n_estimators=100,
                     max_depth=5,
                     max_leaves=64,
                     eta=0.1,
                     reg_lambda=0,
                     tree_method='hist',
                     eval_metric='logloss',
                     use_label_encoder=False,
                     random_state=1000,
                     n_jobs=-1)
start = time.time()
clf.fit(X_train,y_train)
elapsed = time.time() - start
print(f'XGB Training ran in {elapsed:.5f} seconds')
y_pred = clf.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test,y_pred):.2f}')