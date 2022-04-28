import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split

def Red(string):
    return '\033[1;31m' + string + '\033[0m'

train = pd.read_csv('LOL_main.csv')

for i in range(10):
    print(Red(f'====={i}====='))
    target='t1_win'
    train = train.iloc[np.random.permutation(len(train))]
    x_columns = [x for x in train.columns if x not in [target]]

    X_train = train[x_columns][:-1001]
    y_train = train[target][:-1001]
    
    X_test = train[x_columns][-1000:]
    y_test = train[target][-1000:]

    gbm0 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
    gbm0.fit(X_train, y_train)
    y_pred = gbm0.predict(X_train)
    y_predprob = gbm0.predict_proba(X_train)[:,1]
    print('===train===')
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))
    y_pred_ = gbm0.predict(X_test)
    y_predprob_ = gbm0.predict_proba(X_test)[:,1]
    print('===test===')
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred_))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob_))
    
