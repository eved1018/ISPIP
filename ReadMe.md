# Meta-DPI: 

<br>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5706324.svg)](https://doi.org/10.5281/zenodo.5706324)



## Written by Evan Edelstein 

## Manuscript by Mordechai Walder , Dr. Raji Viswanathan, Evan Edelstein, Shahar Lazarev, Moshe Carrol 

<br><br>

### Motivation: 
<p>
Identifying protein interfaces is important to learn how proteins interact with their binding partners,
to uncover the regulatory mechanisms to control biological functions and to develop novel therapeutic agents. A
variety of computational approaches have been developed for predicting a protein’s interfacial residues from its
intrinsic features, such as physico-chemical properties of residues, as well as using template-based information
from known interfaces that share high sequence or structure similarity. Methods that rely on features from
templates will not be successful in predicting interfaces when structural homologues with known interfaces are
not available.
</p>

Requirements:

	- python3  (tested with 3.7 and above)

	- packages listed in requirements.txt to install execute 'pip3 install -r requirements.txt' 

	- optional: pymol, dtreeviz and graphviz

Usage: 
	
		1. cd to MetaDPIv2 
		2. execute: python -m metadpi -mode {mode to use} 


Arguments:

	Input/output:
		-if: [str] default: input - Directory containing input data.
		-of: [str] default: output - Directory to place output of MeatDPI.
		-i: [str] default: input.csv - Csv file should be in input folder.
		-cv: [str] default: cv - Directory containing test and train sets for cross-validation. 

		-trainset:[str] default: test_set.txt - Filename containing proteins for models to train on should be in input folder.
		-testset: [str] default: train_set.txt - Filename containing proteins for models to test on should be in input folder.
		-cutoffs: [str] default:'cutoffs.csv' - Filename containing length of interface or precalculated cutoff for each protein. 

		-model_name: [str] default:'model' - Name of models to import/export.

	Mode selection:
		-mode: ['predict', 'test', 'generate','cv','viz'] default: 'predict' - 
			predict: Use pre-trained model in input folder to predict on set.
			generate: Generate a new rf model from a test set without predicting on any data.
			test: Generate a new rf model from a test set and train on a training set.
			viz: Only call the pymol visualization function.
			cv: perform cross-validation and hyperparameter tuning of models on split training set, the best models are then used to predict on a designated testing set. 

	
	Parameters:
		-randomforest_parameter_trees: [integer] default: 10 - Scikit learn 'n_estimators' parameter.
		-random_forest_parameter_depth: [integer] default: None - Scikit learn 'max_depth' parameter.
		-random_forest_parameter_ccp: [float] default: 0.0 - Scikit learn 'ccp_alpha' parameter. (https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).

		-autocutoff: [int] default: 15 - If no cutoff file is used this sets the default interface cutoff value.

	
	Flags: 
		-pymol: Output pymol session and image of protein with experimental and predicted interfaces overladed. 
		-tv: Output svg image of a randomly sampled tree (for large datasets this can take up a huge amount of time and space) see https://github.com/parrt/dtreeviz for details.
		-xg: Include the use of gradient boosting regression model.
		-nn: Include the use of Multi-layer Perceptron regressor model.


Output:

	- results.csv: this file contains the fscore, MCC, Roc AUC and PR AUC for each individual method and model. 
	- roc_model.csv and pr_model.csv: the TRP and FPR by threshold for each individual method and model, can be used to generate specific ROC or PR graphs. 
	- fscore_mcc_by_protein: the individual fscore and mcc for each protein in the test set. 
	- *.joblib: the trained models from a generate, test or cv run. Move these into the input directory to be used with 'predict' mode. 
	-pairtest.csv: Comparison of statistical significance between AUCs.
		- top triangle: difference in pairs of AUCs
		- bottom triangle: log(10) of p-values for the difference in pairs of AUCs.
	- proteins: Directory containing pymol sessions for each protein in the test set.  
	-cvout: Directory containing the best parameters for each model used in the final prediction, as well as the individual metrics over each cross validation step. 
	


Special thanks to:

Dr. Andras Fiser and Dr. Eduardo J Fajardo for insight and guidance. 
<br>
Terence Parr and Prince Grover for use of dtreeviz.

 
	

