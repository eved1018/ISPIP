import joblib
import pandas as pd 
from sklearn.linear_model import LogisticRegression


def logistic_regresion_test_train(test_frame,train_frame,df,feature_cols,annotated_col,output_path_dir,model_name) -> tuple:
    model = LogisticRegression(random_state=0).fit(train_frame[feature_cols],train_frame[annotated_col])
    prediction = model.predict_proba(test_frame[feature_cols])
    y_prob_interface = [p[1] for p in prediction]    
    test_frame["logisticregresion"] = y_prob_interface
    df['logisticregresion'] = y_prob_interface
    joblib.dump(model, f"{output_path_dir}/LG_{model_name}.joblib", compress=3)  # compression is ON!
    return df, test_frame

def logreg_predict_from_trained_model(df,feature_cols,annotated_col,input_folder_path,model_name) -> pd.DataFrame:
    model = joblib.load( f"{input_folder_path}/LG_{model_name}.joblib")
    prediction = model.predict_proba(df[feature_cols])
    y_prob_interface = [p[1] for p in prediction]    
    df["logisticregresion"] = y_prob_interface
    return df

def logreg_generate_model(df,feature_cols,annotated_col,output_path_dir,model_name):
    model = LogisticRegression(random_state=0).fit(df[feature_cols],df[annotated_col])
    joblib.dump(model, f"{output_path_dir}/LG_{model_name}.joblib", compress=3)  # compression is ON!
    return model 
