import argparse
import os
import pathlib

class ArgsContainer:
    def __init__(self,args: argparse.Namespace, folder_path:pathlib.PosixPath) -> None:
        self.args = args
        self.mode = args.modeselection
        self.folder_path  = folder_path
        self.model_name = args.model_name
        self.autocutoff = args.autocutoff
        self.plotmode = args.plotselection
        self.use_test_train_files = True
        self.use_cutoff_from_file = True
        self.save_tree = args.tree_visualization
        self.usepymol = args.protein_visualization
        self.xg = args.xgboost
        self.nn = args.nuarelnet
        self.models_to_use = []
        self.outputfolder = args.outputfolder
        self.inputfolder = args.inputfolder
        self.cvs_path =  os.path.join(folder_path, self.inputfolder, args.cvfoldername)
        self.output_path_dir= os.path.join(folder_path, self.outputfolder)
        self.input_folder_path = os.path.join(folder_path, self.inputfolder)
        self.input_frames_file = os.path.join(folder_path, self.inputfolder, args.inputfile)
        self.test_proteins_file=  os.path.join(folder_path, self.inputfolder, args.testset)
        self.train_proteins_file =  os.path.join(folder_path, self.inputfolder, args.trainset)
        self.cutoff_frame = os.path.join(folder_path, self.inputfolder, args.cutoffs)
        self.rf_params = [args.randomforest_parameter_trees, args.random_forest_parameter_depth, args.random_forest_parameter_ccp]
        self.pymolscriptpath = os.path.join(folder_path, 'scripts', 'visualization', 'pymolviz.py')
        return


 


