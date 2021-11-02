import statsmodels.api as sm 
import numpy as np 
import pandas as pd 


def logistic_regresion_test_train(test_frame,train_frame,df,feature_cols,annotated_col,output_path_dir,model_name) -> tuple:
    x = sm.add_constant(train_frame[feature_cols])
    logit_model=sm.Logit(train_frame[annotated_col],x)
    result=logit_model.fit()
    coefficients = result.params
    vals = []
    for count,predictor in enumerate(feature_cols, start=1):
        v1 = test_frame[predictor]
        v2 = coefficients[count]
        v3 = v1 * v2
        vals.append(v3)
    
    sum_pred = sum(vals)
    val = -1 *(coefficients[0] + sum_pred)
    exponent = np.exp(val)
    pval = (1/(1+exponent))  # type: ignore
    test_frame['logisticregresion'] = list(pval)
    df['logisticregresion'] = pval
    result.save(f"{output_path_dir}/LG_{model_name}.pickle")
    return df, test_frame

def logreg_predict_from_trained_model(df,feature_cols,annotated_col,input_folder_path,model_name) -> pd.DataFrame:
    result = sm.load(f"{input_folder_path}/LG_{model_name}.pickle")
    coefficients = result.params
    vals = []
    for count,predictor in enumerate(feature_cols, start=1):
        v1 = df[predictor]
        v2 = coefficients[count]
        v3 = v1 * v2
        vals.append(v3)
    
    sum_pred = sum(vals)
    val = -1 *(coefficients[0] + sum_pred)
    exponent = np.exp(val)
    pval = (1/(1+exponent)) #type: ignore
    df['logisticregresion'] = list(pval)
    return df

def logreg_generate_model(df,feature_cols,annotated_col,output_path_dir,model_name):
    x = sm.add_constant(df[feature_cols])
    logit_model=sm.Logit(df[annotated_col],x)
    result=logit_model.fit()
    result.save(f"{output_path_dir}/LG_{model_name}.pickle")
