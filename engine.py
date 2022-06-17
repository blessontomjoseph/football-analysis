from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hyperopt import fmin,hp,tpe,Trials
from hyperopt.pyll.base import scope
from sklearn.metrics import accuracy_score
from functools import partial
import datetime
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import confusion_matrix

cols=['ftr','hthg','htag','htr','year','month','day','tpbm']
model=RandomForestClassifier()
train,val=data[data['date']<=datetime.datetime(2015,1,1)] ,data[data['date']> datetime.datetime(2015,1,1)]
train,val = train[cols],val[cols]
t_x,t_y=train.drop(['ftr'],axis=1),train['ftr']
v_x,v_y=val.drop(['ftr'],axis=1),val['ftr']


params = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 100, 1)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 2, 20, 1)),
    'max_features': hp.quniform('max_features', 0.1, 1, 0.1),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])}



def objective(params,trainx,trainy,valx,valy,model):
    model=model.set_params(**params)
    model.fit(trainx,trainy)
    preds=model.predict(valx)
    acc=accuracy_score(preds,valy)
    return (-1*acc)
    
obj_func=partial(objective,trainx=t_x,trainy=t_y,valx=v_x,valy=v_y,model=model)
trials=Trials()
result = fmin( fn=obj_func,
              space=params,
              algo=tpe.suggest,
              trials=trials,
              max_evals=10
                  )

result['n_estimators'] = int(result['n_estimators'])
result['min_samples_split'] = int(result['min_samples_split'])
result['min_samples_leaf'] = int(result['min_samples_leaf'])
result['criterion'] = ['gini', 'entropy'][int(result['criterion'])]

model=model.set_params(**result)
model.fit(t_x,t_y)
folds=KFold()
tr_score=cross_val_score(model,t_x,t_y,scoring='accuracy',cv=folds).mean()
print(tr_score)
preds=model.predict(v_x)
print(accuracy_score(preds,v_y))
print(confusion_matrix(preds,valy))