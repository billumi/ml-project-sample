
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def rf_pipeline_gridsearch(X,y):
    pipe=Pipeline([('scaler',StandardScaler()),('rf',RandomForestClassifier())])
    params={'rf__n_estimators':[50,100]}
    gs=GridSearchCV(pipe,param_grid=params,cv=3); gs.fit(X,y); return gs.best_estimator_
