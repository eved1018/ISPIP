# Evan Edelstein
import argparse
import pathlib
import os
from argscontainer import ArgsContainer
import sys


def userinterface() -> ArgsContainer:
    """
    See ReadMe for details
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile',
                        default='input_data_all.csv', help='file name')
    parser.add_argument('-mode', '--modeselection', choices=['predict', 'test', 'generate', 'cv', 'viz'], default='predict',
                        help="predict: Use pretrained model in input folder to predict on set.\nTest_Train: genrate a new rf model from a test set and train on a training set.\nGenerate:  genrate a new rf model from a test set without predicting on any data.\ncrossvalidateion")
    parser.add_argument('-trainset', default='train_set.txt', help='')
    parser.add_argument('-testset', default='test_set.txt', help='')
    parser.add_argument('-randomforest_parameter_trees', default=100, help='')
    parser.add_argument('-random_forest_parameter_depth',
                        default=None, help='')
    parser.add_argument('-random_forest_parameter_ccp', default=0.0, help='')
    parser.add_argument('-tv', '--tree_visualization',
                        action='store_true', help='add -tv to use it')
    parser.add_argument('-xg', '--xgboost',
                        action='store_true', help='add -xg to use it')
    parser.add_argument('-nn', '--nuarelnet',
                        action='store_true', help='add -nn to use it, Feature is IN DEVELOPMENT ')
    parser.add_argument('-pymol', '--protein_visualization',
                        action='store_true', help='add -pymol to use it ')
    parser.add_argument('-cutoffs', default='cutoffs.csv', help='')
    parser.add_argument('-autocutoff', default='15', help='')
    parser.add_argument('-model_name', default='model', help='')
    parser.add_argument('-of', '--outputfolder', default='output', help='')
    parser.add_argument('-if', '--inputfolder', default='input', help='')
    parser.add_argument('-cv', '--cvfoldername', default='cv', help='')
    # TODO add this argscontainer
    parser.add_argument('-plot', '--plotselection', choices=[
                        'plot', 'csv', 'both'], default='both', help="output pr and roc curve as csv, png or both")

    args: argparse.Namespace = parser.parse_args()
    args_container: ArgsContainer = parse(args)
    return args_container


def parse(args: argparse.Namespace) -> ArgsContainer:
    folder_path: pathlib.Path = pathlib.Path(__file__).parent.parent
    args_container = ArgsContainer(args, folder_path)

    if args_container.mode == 'test':
        if (not os.path.isfile(args_container.test_proteins_file)) or (not os.path.isfile(args_container.train_proteins_file)):
            print(
                "train and or test sets are not set, random 80/20 ditribution will be used")
            args_container.use_test_train_files = False

    if not os.path.isfile(args_container.cutoff_frame):
        print(
            f"cutoffs not found, a global cutoff of {args.autocutoff} residues will be used (this value can be changed with the -autocutoff flag")
        args_container.use_cutoff_from_file = False

    elif not os.path.isfile(args_container.input_frames_file):
        print("please include an input csv file")
        sys.exit(0)
    if args_container.nn:
        args_container.models_to_use.append("nueralnet")
    if args_container.xg:
        args_container.models_to_use.append("xgboost")

    return args_container
