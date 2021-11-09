
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV,cross_validate
from itertools import chain
import pandas as pd 

"""
currently scorer is roc_auc try adding average_precision as well
"""

def hyperparamtertuning_and_crossvalidation(df:pd.DataFrame, cvs,feature_cols, annotated_col):
    df.reset_index(level=0, inplace=True)
    CViterator = []
    for c,testk in enumerate(cvs):
        traink = cvs[:c] + cvs[c+1:] #train is k-1, test is k 
        traink = list(chain.from_iterable(traink))
        trainIndices = df[df["protein"].isin(traink)].index.values.astype(int) 
        testIndices = df[df["protein"].isin(testk)].index.values.astype(int) 
        CViterator.append((trainIndices, testIndices))  

    p_grid = {"n_estimators": [10,50,100,200], "max_depth": [None,5,10,15], "ccp_alpha":[0.0, 0.25, 0.5, 0.75], "bootstrap":[True, False]}
    rf_model = GridSearchCV(estimator=RandomForestClassifier(), param_grid=p_grid, cv=CViterator, scoring="roc_auc").fit(df[feature_cols],df[annotated_col]).best_estimator_
    logit_models = LogisticRegressionCV(cv=CViterator, scoring="roc_auc").fit(df[feature_cols],df[annotated_col])
    linmodel_frame = pd.DataFrame(cross_validate(LinearRegression(), df[feature_cols],df[annotated_col], cv=CViterator, return_estimator = True, scoring="roc_auc"))
    linear_model = linmodel_frame.loc[linmodel_frame['test_score'].idxmax(),"estimator"]
    param_grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,1)],
          'activation': ['relu','tanh','logistic'],
          'alpha': [0.0001, 0.05],
          'learning_rate': ['constant','adaptive'],
          'solver': ['adam']}

    NN_model = GridSearchCV(estimator=MLPRegressor(), param_grid=param_grid, cv=CViterator, scoring="roc_auc").fit(df[feature_cols],df[annotated_col]).best_estimator_

    return  rf_model,linear_model,logit_models,NN_model

