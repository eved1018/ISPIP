from typing import Tuple
from matplotlib.pyplot import annotate
from sklearn.metrics import auc, matthews_corrcoef, f1_score, precision_recall_curve,roc_curve
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor


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
            results.append(return_val[0])
            roc_curve_data.append(return_val[1])
            pr_curve_data.append(return_val[2])
            test_frame[f'{return_val[3][0]}_bin'] = return_val[3][1]
            fscore_mcc_by_protein[[f'{return_val[3][0]}_fscore', f'{return_val[3][0]}_mcc']] = return_val[4].values.tolist()


    result_df = pd.DataFrame(results, columns = ['predictor','f-score','mcc','roc_auc','pr_auc'])
    return  result_df, roc_curve_data, pr_curve_data , test_frame, fscore_mcc_by_protein

def cutoff_file_parser(cutoff_frame) -> dict:
    cutoff_frame_df :pd.DataFrame  = pd.read_csv(cutoff_frame)  # type: ignore
    cutoff_frame_df.columns= cutoff_frame_df.columns.str.lower()
    cutoff_dict =  dict(zip(cutoff_frame_df['protein'], cutoff_frame_df['cutoff res']))
    return cutoff_dict



def analyses(params) -> list:
    pred, cutoff_dict, test_frame, annotated_col = params
    top = test_frame.sort_values(by=[pred], ascending = False).groupby((["protein"])).apply(lambda x: x.head(cutoff_dict[x.name])).index.get_level_values(1).tolist()
    
    test_frame[f'{pred}_bin'] = [1 if i in top else 0 for i in test_frame.index.tolist()]
    fscore_mcc_per_protein = test_frame.groupby((["protein"])).apply(lambda x: fscore_mcc(x, annotated_col, pred))

    fpr, tpr ,roc_thresholds, roc_auc, precision, recall, pr_thresholds, pr_auc = roc_pr(test_frame, annotated_col, pred)

    fscore, mcc  = fscore_mcc(test_frame, annotated_col, pred)
    
    return_values = [[pred, fscore, mcc, roc_auc, pr_auc],[pred,fpr,tpr,roc_auc,roc_thresholds],[pred,recall,precision, pr_auc, pr_thresholds], [pred, test_frame[f'{pred}_bin']],fscore_mcc_per_protein]
    return return_values



def fscore_mcc(x, annotated_col, pred):
    mcc = matthews_corrcoef(x[annotated_col],x[f'{pred}_bin'])
    fscore = f1_score(x[annotated_col],x[f'{pred}_bin'])
    return fscore , mcc

def roc_pr(x, annotated_col, pred):
    fpr, tpr ,roc_thresholds= roc_curve(x[annotated_col], x[pred])
    roc_auc = round(auc(fpr, tpr),3)
    precision, recall, pr_thresholds = precision_recall_curve(x[annotated_col], x[pred])
    pr_auc = round(auc(recall, precision),3)
    return fpr, tpr ,roc_thresholds, roc_auc, precision, recall, pr_thresholds, pr_auc
