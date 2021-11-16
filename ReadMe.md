Meta Dpi: 


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5706324.svg)](https://doi.org/10.5281/zenodo.5706324)


Written by  Evan Edelstein
 	
requirements:

	-python3
	-packages listed in requirements.txt to install execute 'pip3 install -r requirements.txt'
	- optinoal: pymol
Usage: 

	1) cd to MetaDPIv2 
	2) execute 'python -m metadpi -mode {mode to use}' 

arguments:

	-i: [str] defualt:'input.csv- csv file should be in input folder

	-mode: ['predict', 'test', 'generate'] defualt:'predict' - 
		predict: Use pretrained model in input folder to predict on set.
		test: genrate a new rf model from a test set and train on a training set.
		Generate:  genrate a new rf model from a test set without predicting on any data.
	-model_name: [str] defualt:'model' - name of rf and lg models to import/export 


	-trainset:[str] defualt: test_set.txt - filename containing proteins for models to train on should be in input folder.
	-testset: [str] defualt: train_set.txt - filename containing proteins for models to test on should be in input folder.
	-cutoffs: [str] defualt:'cutoffs.csv' - filename containing length of interface or precalculated cutoff for each protein. 
	-autocutoff: [int] defualt: 15 - if no cutoff file is used this sets the defualt interface cutoff value.	
	
	-randomforest_parameter_trees: [integer] defualt:10 - scikit learn 'n_estimators' parameter.
	-random_forest_parameter_depth: [integer] defualt:None - scikit learn 'max_depth' parameter.
	-random_forest_parameter_ccp: [float] defualt:0.0 - scikit learn 'ccp_alpha' parameter. (https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning)
	-tree_visualization: [bool] defualt:False: output svg image of a randomly sampled tree (for large datasets this can take up a huge amount of time and space) see https://github.com/parrt/dtreeviz for details

	-protein_visualization [bool] defualt:False: output pymol session and image of protein with expermintal and predicted interfaces overlayed. 
	

Special thanks to:
	Mordichia Walder and Shahar Lazaruz for being amazing partners in putting this all together.	
	Dr. Raji Viswanathan for leading this project.
	Dr. Andras Fiser and Dr. Eduardo J Fajardo for insight and guidance. 
	Terence Parr and Prince Grover for use of dtreeviz.
	The current Viswanathan lab for listening to me lectrure about proper python syntax for hours.
	The abishar, without you none of this. 
	

