import pandas as pd 
import joblib


def predict(df,feature_cols,input_folder_path,model_name, models) -> pd.DataFrame:

    if  models is not None:
        rf_model ,linreg_model, logreg_model = models
    else:
        rf_model =  joblib.load(f"{input_folder_path}/RF_{model_name}.joblib")
        logreg_model = joblib.load(f"{input_folder_path}/Logit_{model_name}.joblib")
        linreg_model = joblib.load(f"{input_folder_path}/LinRegr_{model_name}.joblib")


    df:pd.DataFrame = randomforest_predict_from_trained_model(df,feature_cols,rf_model)
    df:pd.DataFrame = logreg_predict_from_trained_model(df,feature_cols,logreg_model)
    df:pd.DataFrame = linreg_predict_from_trained_model(df, feature_cols,linreg_model)
    return df

def randomforest_predict_from_trained_model(df: pd.DataFrame,feature_cols,rf_model) -> pd.DataFrame:
    X_test = df[feature_cols]
    y_prob = rf_model.predict_proba(X_test)
    y_prob_interface = [p[1] for p in y_prob]
    df['randomforest'] = y_prob_interface
    return df 


def logreg_predict_from_trained_model(df,feature_cols,logreg_model) -> pd.DataFrame:
    X_test = df[feature_cols]
    prediction = logreg_model.predict_proba(X_test)  
    df["logisticregresion"] = [p[1] for p in prediction]
    return df

def linreg_predict_from_trained_model(df, feature_cols,linreg_model) -> pd.DataFrame:
    X_test = df[feature_cols]
    prediction = linreg_model.predict(X_test)  
    df["linearregression"] = prediction
    return df