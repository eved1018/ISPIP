import argparse
import pathlib
import pandas as pd
import os 
from .containers.filepaths import FilePaths


def interface() -> FilePaths :
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--inputfile' ,default='input.csv', help= 'file name')
    parser.add_argument('-m','--modeselection' ,choices=['predict', 'test', 'generate'] , default='predict',help= "predict: Use pretrained model in input folder to predict on set.\nTest_Train: genrate a new rf model from a test set and train on a training set.\nGenerate:  genrate a new rf model from a test set without predicting on any data.")
    parser.add_argument('-trainset', default='train_set.txt', help='')
    parser.add_argument('-testset', default='test_set.txt', help='')
    parser.add_argument('-randomforest_parameter_trees', default=10, help='')
    parser.add_argument('-random_forest_parameter_depth', default=None, help='')
    parser.add_argument('-random_forest_parameter_ccp', default=0.0, help='')
    parser.add_argument('-tree_visualization', default=False, help='')
    parser.add_argument('-protein_visualization', default=False, help='')
    parser.add_argument('-cutoffs', default='cutoffs.csv', help='')
    parser.add_argument('-model_name', default='model', help='')
    args = parser.parse_args()
    args_container:FilePaths = parse(args)
    return args_container

def parse(args:argparse.Namespace) -> FilePaths:
    folder_path:pathlib.PosixPath = pathlib.Path(__file__).parent.parent
    args_container = FilePaths(args, folder_path)
    if args_container.mode == 'test':        
        if (not os.path.isfile(args_container.test_proteins_file)) or (not os.path.isfile(args_container.train_proteins_file)):
            print("train and or test sets are not set, random 80/20 ditribution will be used")
            args_container.use_test_train_files = False

    if not os.path.isfile(args_container.cutoff_frame):
        print("cutoffs not found, a global cutoff of 15 residues will be used")
        args_container.use_cutoff_from_file = False

    if not os.path.isfile(args_container.input_frames_file):
        print("please include an input csv file")
        return

    return args_container
    

   
