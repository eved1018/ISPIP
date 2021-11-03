from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, cross_val_score, 



def cross_validation(train_frame ,feature_cols, annotated_col):
    # linear_model = LinearRegressionCV()
    # linear_result = cross_validate(linear_model , train_frame[feature_cols], train_frame[annotated_col], cv = 10, scoring="r2", return_estimator = True, return_train_score= True)
    # print("\n\n",linear_result,"\n\n")
    
        
    # train_frame["linearregression"] = linear_result

    # logistic_model = LinearRegression()
    # logistic_result = cross_validate(logistic_model , train_frame[feature_cols], train_frame[annotated_col], cv = 10, scoring="roc_auc", return_estimator = True)
    # train_frame["logisticregresion"] = logistic_result

    # print(train_frame)

    # LogisticRegressionCV
    return train_frame

