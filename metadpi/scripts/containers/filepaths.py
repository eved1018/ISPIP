from argparse import Namespace
import argparse
import os
import pathlib

class FilePaths:
    def __init__(self,args: argparse.Namespace, folder_path:pathlib.PosixPath) -> None:
        self.args = args
        self.mode = args.modeselection
        self.folder_path  = folder_path
        self.model_name = args.model_name
        self.autocutoff = args.autocutoff
        self.plotmode = args.plotselection
        self.use_test_train_files = True
        self.use_cutoff_from_file = True
        self.output_path_dir= os.path.join(folder_path, 'output')
        self.input_folder_path = os.path.join(folder_path, 'input')
        self.input_frames_file = os.path.join(folder_path, 'input', args.inputfile)
        self.test_proteins_file=  os.path.join(folder_path, 'input', args.testset)
        self.train_proteins_file =  os.path.join(folder_path, 'input', args.trainset)
        self.cutoff_frame = os.path.join(folder_path, 'input', args.cutoffs)
        self.rf_params = [args.randomforest_parameter_trees, args.random_forest_parameter_depth, args.random_forest_parameter_ccp]
        return
 


