from typing import Tuple
from numpy.lib.function_base import append
from pandas.io.stata import StataStrLWriter
from sklearn.metrics import auc, matthews_corrcoef, f1_score, precision_recall_curve,roc_curve
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np 
from .compare_auc_delong_xu import delong_roc_test
import itertools


""""
TODO:
change pval to not log(pval);
generate table:
    for each model:
        for each fold:
            col1=> parameters
            col2=> roc-auc
"""

def postprocess(test_frame,predicted_col,args_container,annotated_col,autocutoff) -> Tuple[pd.DataFrame, list, list]:   
    proteins = test_frame.protein.unique()
    results = []
    roc_curve_data :list = []
    pr_curve_data:list = []
    fscore_mcc_by_protein = pd.DataFrame(index = proteins)
    cutoff_dict = cutoff_file_parser(args_container.cutoff_frame) if (args_container.use_cutoff_from_file) else  {protein:autocutoff for protein in proteins}
    params = [(pred,cutoff_dict, test_frame,annotated_col) for pred in predicted_col] #make testframe and cutoff dict this self in class
   
    with ProcessPoolExecutor(max_workers=4) as exe:
        return_vals = exe.map(analyses,params)
        for return_val in return_vals:
            test_frame[f'{return_val[0]}_bin'] = return_val[1]
            fscore_mcc_by_protein[[f'{return_val[0]}_fscore', f'{return_val[0]}_mcc']] = return_val[2].values.tolist()
            
            results.append(return_val[3])
            roc_curve_data.append(return_val[4])
            pr_curve_data.append(return_val[5])
    
    result_df = pd.DataFrame(results, columns = ['predictor','f-score','mcc','roc_auc','pr_auc'])
    stats_df = pd.DataFrame(index=predicted_col, columns=predicted_col)
    test_frame = test_frame.sort_values(by=annotated_col, ascending = False)
    for index in  predicted_col:
        for column in predicted_col:
            if index == column:
                stats_df.loc[index, column] = index
            else:
                pval,test, auc_diff = statistics(test_frame, annotated_col, index, column) 
                stats_df.loc[index,column] = pval
                stats_df.loc[column,index] = auc_diff
                

    return  result_df, roc_curve_data, pr_curve_data , test_frame, fscore_mcc_by_protein,stats_df

def cutoff_file_parser(cutoff_frame) -> dict:
    cutoff_frame_df :pd.DataFrame  = pd.read_csv(cutoff_frame)  
    cutoff_frame_df.columns= cutoff_frame_df.columns.str.lower()
    cutoff_dict =  dict(zip(cutoff_frame_df['protein'], cutoff_frame_df['cutoff res']))
    return cutoff_dict

def analyses(params) -> list: #TODO make this much better
    pred, cutoff_dict, test_frame, annotated_col = params
    top = test_frame.sort_values(by=[pred], ascending = False).groupby((["protein"])).apply(lambda x: x.head(cutoff_dict[x.name])).index.get_level_values(1).tolist()
    
    test_frame[f'{pred}_bin'] = [1 if i in top else 0 for i in test_frame.index.tolist()]
    fscore_mcc_per_protein = test_frame.groupby((["protein"])).apply(lambda x: fscore_mcc(x, annotated_col, pred))
    
    fscore, mcc = fscore_mcc(test_frame, annotated_col, pred)
    roc_dict  = roc_pr(test_frame, annotated_col, pred)
    
    results_list = [pred, fscore, mcc, roc_dict["roc_auc"], roc_dict["pr_auc"]]
    roclist = [pred, roc_dict["fpr"], roc_dict["tpr"],
               roc_dict["roc_auc"], roc_dict["roc_thresholds"]]

    prlist = [pred,roc_dict["recall"],roc_dict["precision"], roc_dict["pr_auc"], roc_dict["pr_thresholds"]]
    
    return pred,test_frame[f'{pred}_bin'], fscore_mcc_per_protein,results_list,roclist, prlist


def fscore_mcc(x, annotated_col, pred):
    return f1_score(x[annotated_col],x[f'{pred}_bin']) ,matthews_corrcoef(x[annotated_col],x[f'{pred}_bin'])

def statistics(x, annotated_col, pred1, pred2):
    y_true = x[annotated_col]
    y1 = x[pred1]
    y2 = x[pred2]
    pval,aucs  = delong_roc_test(y_true, y1, y2)
    aucs = aucs.tolist()
    Dauc = round(aucs[1] - aucs[0], 3)
    pval = round(pval.tolist()[0][0],3)
    test = "signifigant" if pval < 0.05 else "not significant"
    return pval, test,Dauc
    

def roc_pr(x, annotated_col, pred):
    fpr, tpr ,roc_thresholds = roc_curve(x[annotated_col], x[pred])
    roc_auc = round(auc(fpr, tpr),3)
    precision, recall, pr_thresholds = precision_recall_curve(x[annotated_col], x[pred])
    pr_auc = round(auc(recall, precision),3)
    result_dic = {"fpr":fpr, "tpr":tpr ,"roc_thresholds":roc_thresholds, "roc_auc":roc_auc, "precision":precision, "recall":recall, "pr_thresholds":pr_thresholds, "pr_auc":pr_auc}
    return result_dic
