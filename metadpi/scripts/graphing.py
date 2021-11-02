import  matplotlib.pyplot as plt 
from dtreeviz.trees import *
import pandas as pd 



def roc_viz(roc_curve_data, output_path_dir, model_name):
    roc_frame = pd.DataFrame()

    plt.figure()
    lw = 2
    for data in roc_curve_data:
        pred,fpr,tpr,roc_auc,thresholds = data
        roc_frame[f"{pred}_fpr"] = fpr 
        roc_frame[f"{pred}_tpr"] = tpr 
        plt.plot(fpr,tpr,lw=lw, label=f"{pred} (area = {roc_auc})")


    roc_frame.to_csv(f"{output_path_dir}/roc_{model_name}.csv")
    plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--", alpha = 0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_path_dir}/ROC_{model_name}.png")
    # plt.show()
    return
    
def pr_viz(pr_curve_data, output_path_dir, model_name, df, annotated_col):
    pr_frame = pd.DataFrame()
    plt.figure()
    lw = 2
    for data in pr_curve_data:
        pred,recall,precision,pr_auc,thresholds = data
        pr_frame[f"{pred}_fpr"] = recall 
        pr_frame[f"{pred}_tpr"] = precision 
        plt.plot(recall,precision,lw=lw, label=f"{pred} (area = {pr_auc})")
    pr_frame.to_csv(f"{output_path_dir}/roc_{model_name}.csv")
    no_skill = len(df[df[annotated_col]==1]) / len(df)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{output_path_dir}/PR_{model_name}.png")
    # plt.show()
    return


def treeviz(treeparams,result_path,code,cols):
    (X, y, tree,depth) = treeparams
    viz = dtreeviz(tree, 
        X, 
        y,
        target_name='Interface',
        feature_names= cols, 
        class_names= ["non_interface", "interface"], 
        show_node_labels= True, 
        fancy=False 
        )  
    
    path = f"{result_path}/META_DPI_RESULTS{code}/Trees/Rftree_{depth}.svg" 
    viz.save(path)
    return

def roc_to_csv(roc_curve_data, output_path_dir, model_name):
    roc_frame = pd.DataFrame()
    for data in roc_curve_data:
        pred,fpr,tpr,roc_auc,thresholds = data
        roc_frame[f"{pred}_fpr"] = fpr 
        roc_frame[f"{pred}_tpr"] = tpr 
   
    roc_frame.to_csv(f"{output_path_dir}/roc_{model_name}.csv")
    return

def pr_to_csv(pr_curve_data, output_path_dir, model_name):
    pr_frame = pd.DataFrame()
    for data in pr_curve_data:
        pred,recall,precision,pr_auc,thresholds = data
        pr_frame[f"{pred}_fpr"] = recall 
        pr_frame[f"{pred}_tpr"] = precision 
    pr_frame.to_csv(f"{output_path_dir}/roc_{model_name}.csv")
    return


