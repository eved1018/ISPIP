import pandas as pd 
import  matplotlib.pyplot as plt 


    
def roc_viz(roc_curve_data, output_path_dir, model_name):
    plt.figure()
    lw = 2
    for data in roc_curve_data:
        pred,fpr,tpr,roc_auc,thresholds = data
        plt.plot(fpr,tpr,lw=lw, label=f"{pred} (area = {roc_auc})")


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
    plt.figure()
    lw = 2
    for data in pr_curve_data:
        pred,recall,precision,pr_auc,thresholds = data
        plt.plot(recall,precision,lw=lw, label=f"{pred} (area = {pr_auc})")

    no_skill = len(df[df[annotated_col]==1]) / len(df)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray')
    # plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--", alpha = 0.5)
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