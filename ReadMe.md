Meta Dpi: 

Written by  Evan Edelstein
 	
Usage: 
	1) cd to MetaDpiNew
	2) execute 'python -m metadpi'

arguments:
	-i: [str] defualt:'input.csv- csv file should be in input folder

	-m: mode: ['predict', 'test', 'generate'] defualt:'predict' - 
		predict: Use pretrained model in input folder to predict on set.
		test: genrate a new rf model from a test set and train on a training set.
		Generate:  genrate a new rf model from a test set without predicting on any data.

	-trainset:[str]- filename containing proteins for models to train on should be in input folder.
	-testset: [str] defualt: - filename containing proteins for models to test on should be in input folder.

	-randomforest_parameter_trees: [integer] defualt:10 - scikit learn 'n_estimators' parameter.
	-random_forest_parameter_depth: [integer] defualt:None - scikit learn 'max_depth' parameter.
	-random_forest_parameter_ccp: [float] defualt:0.0 - scikit learn 'ccp_alpha' parameter. (https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning)
	-tree_visualization: [bool] defualt:False: output svg image of a randomly sampled tree (for large datasets this can take up a huge amount of time and space) see https://github.com/parrt/dtreeviz for details
	


Special thanks to:
	Mordichia Walder and Shahar Lazaruz for being amazing partners in putting this all together.	
	Dr. Raji Viswanathan for leading this project.
	Dr. Andras Fiser and Dr. Eduardo J Fajardo for insight and guidance. 
	Terence Parr and Prince Grover for use of dtreeviz.
	The current Viswanathan lab for listening to me lectrure about proper python syntax for hours.
	The abishar, without you none of this. 
	

