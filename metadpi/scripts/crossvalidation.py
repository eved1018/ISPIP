
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd 
from sklearn.metrics import f1_score


def cross_validation(cvs ,feature_cols, annotated_col,args_container, df,pred):
    

    autocutoff = 15
    cutoff_dict = cutoff_file_parser(args_container.cutoff_frame) if (args_container.use_cutoff_from_file) else  {protein:autocutoff for protein in df.protein.unique()}
    params = []

    for c,test in enumerate(cvs): #make kfold splits
        train = cvs[:c] + cvs[c+1:] #train is k-1, test is k 
        train = pd.concat(train)
        params.append((train, test , feature_cols, annotated_col,pred)) 

    
    models = []

    with ProcessPoolExecutor(max_workers=4) as exe:
        results = exe.map(generate, params)
        for i in results:
            model_coefs, model_params,model, test = i
            models.append((test, pred, cutoff_dict,annotated_col,model_coefs, model_params,model))


    cv_list = []

    with ProcessPoolExecutor(max_workers=4) as exe:
        results = exe.map(score, models)
        for i in results:
            cv_list.append(i)
    
    cv_frame = pd.DataFrame(cv_list, columns=['fscore','coefs','params','model'])
    best_model = cv_frame.loc[cv_frame['fscore'].idxmax()]
    model = best_model['model']
    df = predict_from_best_model(df,feature_cols,model,pred)
    fscore, model_coefs, model_params, model = score((df, pred, cutoff_dict,annotated_col ,None, None,None))
    print(f'{pred}: {fscore}')
    return

def generate(params):
    train, test,feature_cols, annotated_col,pred = params
    if pred == "logisticregression":
        model = LogisticRegression(random_state=0).fit(train[feature_cols],train[annotated_col])
        model_params = model.get_params()
        model_coefs = model.coef_
        prediction = model.predict_proba(test[feature_cols])
        y_prob_interface = [p[1] for p in prediction] 
        test["logisticregression"] = y_prob_interface


    elif pred == 'linearregression':
        model = LinearRegression().fit(train[feature_cols],train[annotated_col])
        model_params = model.get_params()
        model_coefs = model.coef_
        y_prob_interface = model.predict(test[feature_cols])
        test["linearregression"] = y_prob_interface

    return model_coefs, model_params, model, test

def score(param):
    test, pred, cutoff_dict,annotated_col ,model_coefs, model_params,model = param
    top = test.sort_values(by=[pred], ascending = False).groupby((["protein"])).apply(lambda x: x.head(cutoff_dict[x.name])).index.get_level_values(1).tolist()
    test[f'{pred}_bin'] = [1 if i in top else 0 for i in test.index.tolist()]
    y_true = test[annotated_col]
    fscore = f1_score(y_true,test[f'{pred}_bin'])
    return fscore, model_coefs, model_params, model

def cutoff_file_parser(cutoff_frame) -> dict:
    cutoff_frame_df : pd.DataFrame= pd.read_csv(cutoff_frame)  # type: ignore
    cutoff_frame_df.columns= cutoff_frame_df.columns.str.lower()
    cutoff_dict =  dict(zip(cutoff_frame_df['protein'], cutoff_frame_df['cutoff res']))
    return cutoff_dict

def predict_from_best_model(df,feature_cols,model, pred):
    if pred == "logisticregression":
        prediction = model.predict_proba(df[feature_cols])  
        df["logisticregression"] = [p[1] for p in prediction] 
    elif pred == 'linearregression':
        df["linearregression"] = model.predict(df[feature_cols]) 
    return df

# def predict_from_avrg(coefficients):
#     vals = []
#     for count,predictor in enumerate(feature_cols, start=1):
#         v1 = df[predictor]
#         v2 = coefficients[count]
#         v3 = v1 * v2
#         vals.append(v3)
    
#     sum_pred = sum(vals)
#     val = -1 *(coefficients[0] + sum_pred)
#     exponent = np.exp(val)
#     pval = (1/(1+exponent))
#     df['logisticregresion'] = list(pval)