from sklearn import linear_model
import joblib

def linear_regresion_test_train(test_frame, train_frame, df, feature_cols, annotated_col,output_path_dir,model_name ):
    regr = linear_model.LinearRegression()
    regr.fit(train_frame[feature_cols], train_frame[annotated_col])
    prediction = regr.predict(test_frame[feature_cols])
    test_frame["linearregression"] = prediction
    df["linearregression"] = test_frame["linearregression"]
    joblib.dump(regr, f"{output_path_dir}/LinRegr_{model_name}.joblib", compress=3)  # compression is ON!
    return df, test_frame

def linreg_predict_from_trained_model(df, feature_cols, input_folder_path,model_name):
    X_test = df[feature_cols]
    regr = joblib.load(f"{input_folder_path}/RF_{model_name}.joblib")
    prediction = regr.predict(X_test)  
    df["linearregression"] = prediction
    return df

def linreg_generate_model(df, feature_cols, annotated_col, output_path_dir, model_name):
    regr = linear_model.LinearRegression().fit(df[feature_cols], df[annotated_col])
    joblib.dump(regr, f"{output_path_dir}/LinRegr_{model_name}.joblib", compress=3)  # compression is ON!
    return
