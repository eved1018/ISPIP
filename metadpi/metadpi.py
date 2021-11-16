import os
from .scripts.userinterface import userinterface
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from .scripts.postprocess import postprocess
from .scripts.crossvalidation import hyperparamtertuning_and_crossvalidation
from .scripts.generate import generate
from .scripts.predict import predict
from .scripts.graphing import roc_viz, pr_viz, treeviz, pymol_viz
from .scripts.preprocess import data_preprocesss, data_split_auto,data_split_from_file,cross_validation_set_generater

"""
TODO:
1) change return type tuple to be Tuple[type1,type2...] ad inlude typesaftey on arguments!
2) for output dir: create nested folders for ease of acess!
"""

def main() -> None:
    args_container = userinterface()         # get Command line arguments and defualts 
    df = pd.read_csv(args_container.input_frames_file) # index_col = 0, wrap this in a try statement    #read input file containing residues, individual predictors and annotated columns
    df, feature_cols, annotated_col, proteins  = data_preprocesss(df)   #preprocess data -> remove any null or missing data from the dataset and check that annoted is number  nulls 
    predicted_col = feature_cols + ['logisticregresion', "linearregression",'randomforest','nueralnet','xgboost']     # make this automatic? if we do more models later change this!!! 

    #Mode 1: predict 
    
    if args_container.mode == 'predict':
        df = predict(df,feature_cols,args_container.input_folder_path,args_container.model_name)
        results_df, roc_curve_data, pr_curve_data , bin_frame, stats_df= postprocess(df,predicted_col,args_container,annotated_col,args_container.autocutoff)
        visualization(roc_curve_data,pr_curve_data ,None,df,feature_cols,annotated_col,predicted_col,df,bin_frame,args_container)

    #Mode 2: generate learned model 
    
    elif  args_container.mode == 'generate':
        models,tree = generate(df, feature_cols, annotated_col, args_container.output_path_dir, args_container.model_name,args_container.rf_params )
        if (tree is not None) and args_container.save_tree:    
            treeviz(tree,df,feature_cols,annotated_col, args_container.model_name, args_container.output_path_dir)

    #Mode 3: test/train

    elif args_container.mode == 'test':

        if args_container.use_test_train_files:
            test_frame, train_frame = data_split_from_file(df, args_container)
        else:
            test_frame, train_frame = data_split_auto(df, proteins)
        
        print(f'lenght of test set: {len(test_frame)}', f"length of training set: {len(train_frame)}")
        models, tree = generate(train_frame, feature_cols, annotated_col, args_container.output_path_dir, args_container.model_name,args_container.rf_params ) #train 
        test_frame  = predict(test_frame,feature_cols,args_container.input_folder_path,args_container.model_name, models) #test
        results_df, roc_curve_data,pr_curve_data , bin_frame, fscore_mcc_by_protein, stats_df= postprocess(test_frame,predicted_col,args_container,annotated_col,args_container.autocutoff)
        df_saver(results_df, "results", args_container.output_path_dir)
        df_saver(bin_frame, "bin_frame", args_container.output_path_dir)
        df_saver(fscore_mcc_by_protein, "fscore_mcc_by_protein", args_container.output_path_dir)
        df_saver(stats_df, "pairtest",args_container.output_path_dir )
        visualization(roc_curve_data,pr_curve_data ,tree,df,feature_cols,annotated_col,predicted_col,test_frame ,bin_frame,args_container)
        print(results_df)


    #Mode 4: Cross-validation: 
    
    elif  args_container.mode == 'cv':
        test_frame, cvs,train_proteins = cross_validation_set_generater(args_container.cvs_path,df)
        models = hyperparamtertuning_and_crossvalidation(df, train_proteins,feature_cols, annotated_col,args_container)
        model_param_writer(models, args_container.output_path_dir)    
        test_frame = predict(test_frame,feature_cols,args_container.input_folder_path,args_container.model_name,  models)
        results_df, roc_curve_data,pr_curve_data , bin_frame, fscore_mcc_by_protein, stats_df= postprocess(test_frame,predicted_col,args_container,annotated_col,args_container.autocutoff)
        df_saver(results_df, "results", args_container.output_path_dir)
        df_saver(bin_frame, "bin_frame", args_container.output_path_dir)
        df_saver(fscore_mcc_by_protein, "fscore_mcc_by_protein", args_container.output_path_dir)
        df_saver(stats_df, "pairtest",args_container.output_path_dir )
        visualization(roc_curve_data,pr_curve_data ,None,df,feature_cols,annotated_col,predicted_col,test_frame ,bin_frame,args_container)
        print(results_df)

    else:
        print("mode is set incorrectly")
    return

def df_saver(df, name, output_path_dir):
    out = os.path.join(output_path_dir, f'{name}.csv')
    df.to_csv(out)
    return

def model_param_writer(models, output_path_dir):
    rf_model, linear_model, logit_model,NN_model ,xgb_model = models 
    out = os.path.join(output_path_dir, f'best_parameters.txt')        
    with open(out,'w+') as file:
        file.write(f"random forest params: {rf_model.get_params()}\nLinear regr coefs: {linear_model.coef_}\nLogit regr coefs:{logit_model.coef_}\nNN params: {NN_model.get_params()}")
    return
    
def visualization(roc_curve_data,pr_curve_data ,tree,df,feature_cols,annotated_col,predicted_col,test_frame, bin_frame,args_container):
    roc_viz(roc_curve_data,args_container.output_path_dir, args_container.model_name)
    pr_viz(pr_curve_data,args_container.output_path_dir,args_container.model_name, test_frame, annotated_col) 
    if (tree is not None) and args_container.save_tree:    
        treeviz(tree,df,feature_cols,annotated_col, args_container.model_name, args_container.output_path_dir)
    protein_to_viz = bin_frame["protein"].unique()[0] #TODO set this as a config (do all or list)
    print(protein_to_viz)
    if args_container.usepymol:
        pymol_viz(bin_frame, protein_to_viz, predicted_col,annotated_col, args_container.pymolscriptpath, args_container.output_path_dir)
    return