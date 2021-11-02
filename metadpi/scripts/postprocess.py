from typing import Tuple
from sklearn.metrics import auc,roc_curve, matthews_corrcoef, f1_score, precision_recall_curve
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import  matplotlib.pyplot as plt 


def postprocess(test_frame,feature_cols,args_container,annotated_col,autocutoff) -> Tuple[pd.DataFrame, list, list]:   
    predicted_col = feature_cols + ['logisticregresion', "linearregrsion",'randomforest']
    results = []
    roc_curve_data :list = []
    pr_curve_data:list = []
    cutoff_dict = cutoff_file_parser(args_container.cutoff_frame) if (args_container.use_cutoff_from_file) else  {protein:autocutoff for protein in test_frame.protein.unique()}
    bin_frame = test_frame
    params = [(pred,cutoff_dict, bin_frame,annotated_col) for pred in predicted_col] #make testframe and cutoff dict this self in class
    with ProcessPoolExecutor(max_workers=4) as exe:
        return_vals = exe.map(analyses,params)
        for return_val in return_vals:
            results.append(return_val[0])
            roc_curve_data.append(return_val[1])
            pr_curve_data.append(return_val[2])

    result_df = pd.DataFrame(results, columns = ['predictor','f-score','mcc','roc_auc','pr_auc'])
    return  result_df, roc_curve_data, pr_curve_data

def cutoff_file_parser(cutoff_frame) -> dict:
    cutoff_frame_df = pd.read_csv(cutoff_frame)
    cutoff_frame_df.columns= cutoff_frame_df.columns.str.lower()
    cutoff_dict =  dict(zip(cutoff_frame_df['protein'], cutoff_frame_df['cutoff res']))
    return cutoff_dict



def analyses(params) -> list:
    pred, cutoff_dict, bin_frame, annotated_col = params
    bin_frame[pred]  = [1 if i in bin_frame.groupby(["protein"]).apply(lambda x: x.nlargest(cutoff_dict[x.name], columns = [pred])).index.get_level_values(1) else 0 for i in bin_frame.index.values.tolist()]
    y_true = bin_frame[annotated_col]
    y_pred = bin_frame[pred]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, drop_intermediate=False)
    roc_auc = round(auc(fpr, tpr),3)
    mcc = matthews_corrcoef(y_true,y_pred, sample_weight=None)
    fscore = f1_score(y_true,y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = round(auc(recall, precision),3)
    return_values = [[pred, fscore, mcc, roc_auc, pr_auc],[pred,fpr,tpr,roc_auc,thresholds],[pred,recall,precision, pr_auc, thresholds]]
    return return_values
