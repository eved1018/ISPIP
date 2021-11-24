import argparse
import os
import pathlib


class ArgsContainer:
    def __init__(self, args: argparse.Namespace, folder_path: pathlib.PosixPath) -> None:
        self.args: argparse.Namespace = args
        self.mode: str = args.modeselection
        self.folder_path: pathlib.PosixPath = folder_path
        self.model_name: str = args.model_name
        self.autocutoff: int = args.autocutoff
        self.plotmode: str = args.plotselection
        self.use_test_train_files: bool = True
        self.use_cutoff_from_file: bool = True
        self.save_tree: bool = args.tree_visualization
        self.usepymol: bool = args.protein_visualization
        self.xg: bool = args.xgboost
        self.nn: bool = args.nuarelnet
        self.models_to_use: list = []
        self.outputfolder = args.outputfolder
        self.inputfolder = args.inputfolder
        self.cvs_path = os.path.join(
            folder_path, self.inputfolder, args.cvfoldername)
        self.output_path_dir = os.path.join(folder_path, self.outputfolder)
        self.input_folder_path = os.path.join(folder_path, self.inputfolder)
        self.input_frames_file = os.path.join(
            folder_path, self.inputfolder, args.inputfile)
        self.test_proteins_file = os.path.join(
            folder_path, self.inputfolder, args.testset)
        self.train_proteins_file = os.path.join(
            folder_path, self.inputfolder, args.trainset)
        self.cutoff_frame = os.path.join(
            folder_path, self.inputfolder, args.cutoffs)
        self.rf_params = [args.randomforest_parameter_trees,
                          args.random_forest_parameter_depth, args.random_forest_parameter_ccp]
        self.pymolscriptpath = os.path.join(
            folder_path, 'scripts', 'visualization', 'pymolviz.py')
        return

