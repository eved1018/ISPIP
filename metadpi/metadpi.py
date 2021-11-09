import os

from sklearn import linear_model
from .scripts.userinterface import userinterface
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'
from .scripts.postprocess import postprocess
from .scripts.crossvalidation import hyperparamtertuning_and_crossvalidation
from .scripts.generate import generate
from .scripts.predict import predict
from .scripts.graphing import roc_viz, pr_viz, treeviz, pymol_viz


"""
TODO:
1) change return type tuple to be Tuple[type1,type2...]
2) up the oop
3) make the parametrs less terrible!
"""

def run() -> None:
    """ add doctring tests and whatnot"""
    args_container = userinterface()         # get Command line arguments and defualts 
    df = pd.read_csv(args_container.input_frames_file) # index_col = 0, wrap this in a try statement    #read input file containing residues, individual predictors and annotated columns
    df, feature_cols, annotated_col, proteins  = data_preprocesss(df)   #preprocess data -> remove any null or missing data from the dataset and check that annoted is number  nulls 
    predicted_col = feature_cols + ['logisticregresion', "linearregression",'randomforest','nueralnet']     # make this automatic? if we do more models later change this!!! 

    #Mode 1: predict 
    
    if args_container.mode == 'predict':
        df = predict(df,feature_cols,args_container.input_folder_path,args_container.model_name)
        results_df, roc_curve_data, pr_curve_data , bin_frame= postprocess(df,predicted_col,args_container,annotated_col,args_container.autocutoff)
        visualization(roc_curve_data,pr_curve_data ,None,df,feature_cols,annotated_col,predicted_col,df,bin_frame,args_container)

    #Mode 2: generate learned model 
    
    elif  args_container.mode == 'generate':
        models,tree = generate(df, feature_cols, annotated_col, args_container.output_path_dir, args_container.model_name,args_container.rf_params )
        # treeviz(tree,df,feature_cols,annotated_col, args_container.model_name, args_container.output_path_dir)
   

    #Mode 3: test/train

    elif args_container.mode == 'test':

        if args_container.use_test_train_files:
            test_frame, train_frame = data_split_from_file(df, args_container)
        else:
            test_frame, train_frame = data_split_auto(df, proteins)
        
        print(f'lenght of test set: {len(test_frame)}', f"length of training set: {len(train_frame)}")
        
        models, tree = generate(train_frame, feature_cols, annotated_col, args_container.output_path_dir, args_container.model_name,args_container.rf_params ) #train 
        test_frame  = predict(test_frame,feature_cols,args_container.input_folder_path,args_container.model_name, models) #test
        results_df, roc_curve_data,pr_curve_data , bin_frame, fscore_mcc_by_protein= postprocess(test_frame,predicted_col,args_container,annotated_col,args_container.autocutoff)
        df_saver(results_df, "results", args_container.output_path_dir)
        df_saver(bin_frame, "bin_frame", args_container.output_path_dir)
        df_saver(fscore_mcc_by_protein, "fscore_mcc_by_protein", args_container.output_path_dir)
        visualization(roc_curve_data,pr_curve_data ,tree,df,feature_cols,annotated_col,predicted_col,test_frame ,bin_frame,args_container)
        print(results_df)


    #Mode 4: Cross-validation: 
    
    elif  args_container.mode == 'cv':
        test_frame, cvs,train_proteins = cross_validation_set_generater(args_container.cvs_path,df)
        randomforest_model, linear_model,logit_model,NN_model = hyperparamtertuning_and_crossvalidation(df, train_proteins,feature_cols, annotated_col)
        models = [randomforest_model, linear_model, logit_model,NN_model]
        print(f"random forest params: {randomforest_model.get_params()}\nLinear regr coefs: {linear_model.coef_}\nLogit regr coefs:{logit_model.coef_}")
        test_frame = predict(test_frame,feature_cols,args_container.input_folder_path,args_container.model_name,  models)
        results_df, roc_curve_data,pr_curve_data , bin_frame, fscore_mcc_by_protein= postprocess(test_frame,predicted_col,args_container,annotated_col,args_container.autocutoff)
        print(results_df)
        df_saver(results_df, "results", args_container.output_path_dir)
        df_saver(bin_frame, "bin_frame", args_container.output_path_dir)
        df_saver(fscore_mcc_by_protein, "fscore_mcc_by_protein", args_container.output_path_dir)
        visualization(roc_curve_data,pr_curve_data ,None,df,feature_cols,annotated_col,predicted_col,test_frame ,bin_frame,args_container)

    else:
        print("mode is set incorrectly")
    return

def data_preprocesss(df: pd.DataFrame) -> tuple: 
    feature_cols = df.columns.tolist()[1:-1]
    annotated_col = df.columns.tolist()[-1]
    df["protein"] = [x.split('_')[1] for x in df['residue']]
    proteins = df["protein"].unique()
    df.set_index('residue', inplace= True )
    df.isnull().any()
    df = df.fillna(method='ffill')
    df = df[df['annotated'] != "ERROR"]
    df["annotated"] = pd.to_numeric(df["annotated"])
    return df, feature_cols,annotated_col, proteins    
    
def data_split_auto(df, proteins) -> tuple:
    test_set, train_set = train_test_split(proteins,test_size=0.2) 
    train_frame: pd.DataFrame =df[df["protein"].isin(train_set)]
    test_frame: pd.DataFrame= df[df["protein"].isin(test_set)]
    return test_frame, train_frame

def data_split_from_file(df, filepaths) -> tuple: 
    test_file = filepaths.test_proteins_file
    train_file = filepaths.train_proteins_file
    test_frame,lines = input_parser(test_file, df)
    train_frame,lines = input_parser(train_file, df)
    return test_frame, train_frame

def cross_validation_set_generater(cvs_path,df):
    cvs = []
    train_proteins = []
    for file_name in os.listdir(cvs_path):
        if file_name.startswith("training"):
            file_name_path = os.path.join(cvs_path, file_name)
            train_frame,lines = input_parser(file_name_path, df)
            cvs.append(train_frame)
            train_proteins.append(lines)

        elif file_name.startswith("test"):
            file_name_path = os.path.join(cvs_path, file_name)
            test_frame,_ = input_parser(file_name_path, df) 
            #TODO make sure only one test set is included 
        else:
            print("please include test and train sets")
    
    return test_frame, cvs, train_proteins

def input_parser(file, df):
    with open(file) as f:
        lines = f.read().rstrip().splitlines()
    lines = [i if i[-2] == '.' else f"{i[:-1]}.{i[-1:]}"  if i[-2].isdigit() or i[-2].isalpha() else  i.replace(i[-2],'.')  for i in lines]
    frame : pd.DataFrame = df[df['protein'].isin(lines)]
    return frame, lines

def df_saver(df, name, output_path_dir):
    out = os.path.join(output_path_dir, f'{name}.csv')
    df.to_csv(out)
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