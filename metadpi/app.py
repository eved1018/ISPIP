from .scripts.interface import interface
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'
from .scripts.postprocess import postprocess
from .scripts.logisticregrsion import logistic_regresion_test_train, logreg_predict_from_trained_model, logreg_generate_model
from .scripts.randomforest import randomforest_test_train, randomforest_predict_from_trained_model,randomforest_generate_model
from .scripts.graphing import roc_viz, pr_viz, treeviz, pymol_viz
from .scripts.linearregression import linear_regresion_test_train, linreg_predict_from_trained_model,linreg_generate_model
from .scripts.crossvalidation import cross_validation


"""
TODO:
1) change return type tuple to be Tuple[type1,type2...]
2) oop the parts that make sense to oop 
3) streamline mode switching stuff to speed up proccesing (ie different function for each mode)
"""


def run() -> None:
    args_container = interface()
    df = pd.read_csv(args_container.input_frames_file)
    df, feature_cols, annotated_col, proteins  = data_preprocesss(df) 
    predicted_col = feature_cols + ['logisticregresion', "linearregression",'randomforest']


    #mode selection 
    if args_container.mode == 'predict':
        df = linreg_predict_from_trained_model(df, feature_cols, args_container.input_folder_path, args_container.model_name)
        df = logreg_predict_from_trained_model(df,feature_cols,annotated_col,args_container.input_folder_path,args_container.model_name)
        df = randomforest_predict_from_trained_model(df,feature_cols,annotated_col, args_container.rf_params, args_container.input_folder_path,args_container.model_name)
        results_df, roc_curve_data, pr_curve_data , bin_frame= postprocess(df,predicted_col,args_container,annotated_col,args_container.autocutoff)
        roc_viz(roc_curve_data, args_container.output_path_dir,args_container.model_name)
        pr_viz(pr_curve_data,args_container.output_path_dir,args_container.model_name, df, annotated_col)
        protein_to_viz = bin_frame["protein"].unique()[0] #TODO set this as a config (do all or list)
        pymol_viz(bin_frame, protein_to_viz, predicted_col,annotated_col, args_container.pymolscriptpath, args_container.output_path_dir)
        print(results_df)

    elif args_container.mode == 'test':
        if args_container.use_test_train_files:
            test_frame, train_frame = data_split_from_file(df, proteins, args_container)
    
        else:
            test_frame, train_frame = data_split_auto(df, proteins)
        
        df, test_frame = linear_regresion_test_train(test_frame,train_frame,df,feature_cols,annotated_col, args_container.output_path_dir, args_container.model_name)
        df, test_frame = logistic_regresion_test_train(test_frame,train_frame,df,feature_cols,annotated_col, args_container.output_path_dir, args_container.model_name)
        df, test_frame, tree = randomforest_test_train(test_frame,train_frame,df,feature_cols,annotated_col, args_container.rf_params,args_container.output_path_dir,args_container.model_name)

        results_df, roc_curve_data,pr_curve_data , bin_frame= postprocess(test_frame,predicted_col,args_container,annotated_col,args_container.autocutoff)
        roc_viz(roc_curve_data,args_container.output_path_dir, args_container.model_name)
        pr_viz(pr_curve_data,args_container.output_path_dir,args_container.model_name, test_frame, annotated_col)        
        # treeviz(tree,df,feature_cols,annotated_col, args_container.model_name, args_container.output_path_dir)
        # protein_to_viz = bin_frame["protein"].unique()[0] #TODO set this as a config (do all or list)
        # pymol_viz(bin_frame, protein_to_viz, predicted_col,annotated_col, args_container.pymolscriptpath, args_container.output_path_dir)
        print(results_df)
    
    
    elif  args_container.mode == 'generate':
        linreg_generate_model(df, feature_cols, annotated_col, args_container.output_path_dir, args_container.model_name)
        logreg_generate_model(df,feature_cols,annotated_col,args_container.output_path_dir,args_container.model_name)
        tree = randomforest_generate_model(df,feature_cols,annotated_col, args_container.rf_params,args_container.output_path_dir, args_container.model_name)
        treeviz(tree,df,feature_cols,annotated_col, args_container.model_name, args_container.output_path_dir)
   
    elif  args_container.mode == 'cv':
        test_frame, train_frame = data_split_auto(df, proteins)
        cross_validation(train_frame ,feature_cols, annotated_col)

   
    else:
        print("mode is set incorrectly")

    

    return

def data_preprocesss(df: pd.DataFrame) -> tuple: 
    feature_cols = df.columns.tolist()[1:-1]
    annotated_col = df.columns.tolist()[-1]
    df["protein"] = [x.split('_')[1] for x in df['residue']]
    proteins = df["protein"].unique()
    df.set_index('residue', inplace= True )
    # remove any null or missing data from the dataset and check that annoted is number 
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

def data_split_from_file(df, proteins, filepaths) -> tuple: 
    test_file = filepaths.test_proteins_file
    train_file = filepaths.train_proteins_file
    test_frame = input_parser(test_file, df)
    train_frame = input_parser(train_file, df)
    return test_frame, train_frame

def input_parser(file, df):
    with open(file) as f:
        lines = f.read().rstrip().splitlines()

    lines = [i if i[-2] == '.' else f"{i[:-1]}.{i[-1:]}"  if i[-2].isdigit() or i[-2].isalpha() else  i.replace(i[-2],'.')  for i in lines]
    frame : pd.DataFrame = df[df['protein'].isin(lines)]
    return frame

