from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import joblib


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

def randomforest_predict_from_trained_model(df: pd.DataFrame,feature_cols,annotated_col, rf_params, input_folder_path,model_name) -> pd.DataFrame:
    X_test = df[feature_cols]
    loaded_rf = joblib.load(f"{input_folder_path}/RF_{model_name}.joblib")
    y_prob = loaded_rf.predict_proba(X_test)
    y_prob_interface = [p[1] for p in y_prob]
    df['randomforest'] = y_prob_interface
    return df 


def randomforest_generate_model(df: pd.DataFrame,feature_cols,annotated_col, rf_params,output_path_dir, model_name) -> None:
    trees, depth, ccp = rf_params
    print(df)
    X = df[feature_cols]
    y = df[annotated_col]
    model = RandomForestClassifier(n_estimators = trees, random_state = 0, bootstrap=False, max_depth=depth, ccp_alpha= ccp)
    model.fit(X, y)
    tree = model.estimators_[0]
    joblib.dump(model, f"{output_path_dir}/RF_{model_name}.joblib", compress=3)  # compression is ON!
    return tree
