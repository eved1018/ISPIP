from sklearn import linear_model
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

def test_train(test_frame,train_frame,df,feature_cols,annotated_col, rf_params,output_path_dir, model_name):
    df ,test_frame , tree= randomforest_test_train(test_frame,train_frame,df,feature_cols,annotated_col, rf_params,output_path_dir, model_name)
    df,test_frame = linear_regresion_test_train(test_frame, train_frame, df, feature_cols, annotated_col,output_path_dir,model_name )
    df ,test_frame = logistic_regresion_test_train(test_frame, train_frame, df, feature_cols, annotated_col,output_path_dir,model_name )
    return df, test_frame, tree

def randomforest_test_train(test_frame: pd.DataFrame,train_frame: pd.DataFrame,df: pd.DataFrame,feature_cols,annotated_col, rf_params,output_path_dir, model_name) -> tuple:
    trees, depth, ccp = rf_params
    X = train_frame[feature_cols]
    y = train_frame[annotated_col]
    X_test = test_frame[feature_cols]
    model = RandomForestClassifier(n_estimators = trees, random_state = 0, bootstrap=False, max_depth=depth, ccp_alpha= ccp)
    model.fit(X, y)
    tree = model.estimators_[0]
    y_prob = model.predict_proba(X_test)
    y_prob_interface = [p[1] for p in y_prob]
    test_frame['randomforest'] = y_prob_interface
    df['randomforest']  = test_frame['randomforest']
    joblib.dump(model, f"{output_path_dir}/RF_{model_name}.joblib", compress=3)  # compression is ON!
    return df , test_frame , tree


def linear_regresion_test_train(test_frame, train_frame, df, feature_cols, annotated_col,output_path_dir,model_name ):
    regr = linear_model.LinearRegression()
    regr.fit(train_frame[feature_cols], train_frame[annotated_col])
    prediction = regr.predict(test_frame[feature_cols])
    test_frame["linearregression"] = prediction
    df["linearregression"] = test_frame["linearregression"]
    joblib.dump(regr, f"{output_path_dir}/LinRegr_{model_name}.joblib", compress=3)  # compression is ON!
    return df, test_frame


def logistic_regresion_test_train(test_frame, train_frame, df, feature_cols, annotated_col,output_path_dir,model_name ):
    regr = linear_model.LogisticRegression()
    regr.fit(train_frame[feature_cols], train_frame[annotated_col])
    prediction = regr.predict_proba(test_frame[feature_cols])
    test_frame["logisticregresion"] = [p[1] for p in prediction]
    df["logisticregresion"] = test_frame["logisticregresion"]
    joblib.dump(regr, f"{output_path_dir}/Logit_{model_name}.joblib", compress=3)  # compression is ON!
    return df, test_frame