# ISPIP: Integrated Structure-based Protein Interface Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6323262.svg)](https://doi.org/10.5281/zenodo.6323262)

---

<p> Written by Evan Edelstein </p>

<p> Manuscript by R. Viswanathan, M. Walder, E. Edelstein, S. Lazarev, M. Carroll, J.E. Fajardo, A. Fiser </p>

[Walder, M., Edelstein, E., Carroll, M. et al. Integrated structure-based protein interface prediction.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04852-2)

---

### Abstract: 

__Background__
<p>Identifying protein interfaces can inform how proteins interact with their binding partners, uncover the regulatory mechanisms that control biological functions and guide the development of novel therapeutic agents. A variety of computational approaches have been developed for predicting a protein’s interfacial residues from its known sequence and structure. Methods using the known three- dimensional structures of proteins can be template-based or template-free. Template-based methods have limited success in predicting interfaces when homologues with known complex structures are not available to use as templates. The prediction performance of template-free methods that only rely only upon proteins’ intrinsic properties is limited by the amount of biologically relevant features that can be included in an interface prediction model.</p>

__Results__
<p> We describe the development of an integrated method, ISPIP, to explore the hypothesis that the efficacy of a computational prediction method of protein binding sites can be enhanced by using a combination of methods that rely on orthogonal structure-based properties of a query protein, combining and balancing both template-free and template-based features. ISPIP is a method that integrates these approaches through simple linear or logistic regression models and more complex decision tree models. On a diverse test set of 156 query proteins, ISPIP outperforms each of its individual classifiers in identifying protein binding interfaces. </p>

__Conclusions__ 
<p>The integrated method captures the best performance of individual classifiers and delivers an improved interface prediction. The method is robust and performs well even when one of the individual classifiers performs poorly on a particular query protein. This work demonstrates that integrating orthogonal methods that depend on different structural properties of proteins performs better at interface prediction than any individual classifier alone.</p>


---

| ![image](Media/output-onlinegiftools.gif) | ![image](Media/legend-removebg-preview.png) |
| --- | --- |
| The structure of 1CP2.A is shown with the annotated and predicted interface resiues highlighted in pink and green respectively | 

---

<h3> Requirements: </h3>

* python3.7 or Docker

<h3>CLI Usage: </h3>

```shell
pip install ISPIP
ispip -i /path/to/input/file --mode generate
```

<h3>Docker Usage: </h3>

```shell
# Clone repo 
git clone https://github.com/eved1018/ISPIP.git
cd ISPIP

# Build docker image
docker build --rm --pull -f "Dockerfile" -t ispip:latest 

# Run ispip
docker run -v $PWD:$PWD --rm --name ispip_run ispip:latest python /usr/src/ispip/main.py -i /path/to/input/file --mode generate
```


<h3>Development: </h3>

```shell
git clone https://github.com/eved1018/ISPIP
cd ISPIP
pip3 install -r requirements.txt
python3 main.py -i /path/to/input/file
```

<h3>Arguments:</h3>

- Input/Output:
	* `-if`: [str] default: None - Directory containing trained models. This folder should contain .joblib files to use as model inputs. 


			| Model    | Name |
			| -------- | ------- |
			| RandomForest  | RF_{model_name}.joblib    |
			| Log Regression  | Logit_{model_name}.joblib    |
			| Lin Regression  | LinRerg_{model_name}.joblib    |
			| XGBoost  | XGB_{model_name}.joblib    |

	* `-of`: [str] default: output - Directory to place output of ISPIP.
	* `-i`: [str] default: input.csv - CSV Filename with columns: "residue","predus","ispred","dockpred","annotated". The column residue is of the form {residue number}_{PDB ID}.{chain}. The annotated column is 1 or interface residue and 0 for non-interface residue
	* `-cv`: [str] default: cv -Directory containing test and train sets for cross-validation. Same csv format as train/test. Filenames should start with train and test
	* `--trainset`: [str] default: test_set.txt - CSV Filename containing proteins for models to train on with columns: protein,size. The column protein is of the form {PDB ID}.{chain}
	* `--testset`: [str] default: train_set.txt - CSV Filename containing proteins for models to test on with columns: protein,size. The column protein is of the form {PDB ID}.{chain}
	* `--cutoffs`: [str] default:'cutoffs.csv' - CSV Filename containing length of interface or precalculated cutoff for each protein. File should have columns: Protein,surface res,cutoff res,annotated res. 
	* `--model-name`: [str] default:'model' - Name of models to import/export. (see -if about)
	* `--results-df`: [str] - path to result file from previous "predict" run to reprocess. (normally named bin_frame.csv)

- Mode selection:
	* `--mode`: ['predict', 'train', 'generate','cv','viz', "reprocess"] default: 'predict'  
		* __predict__: Use pre-trained model in input folder to predict on set.
		* __generate__: Generate a new rf model from a test set without predicting on any data.
		* __train__: Generate a new rf model from a test set and train on a training set (the runs predict).
		* __viz__: Only call the pymol visualization function. (takes --results_df_input and -cv as input)
		* __cv__: Perform cross-validation and hyperparameter tuning of models on split training set, the best models are then used to predict on a designated testing set.  
		* __reprocess__: Generate statistics from a succesful predict run. (takes --results_df_input as input)

- Parameters: 
	* `--rf-trees`: [integer] default: 10 - Scikit learn 'n_estimators' parameter.
	* `--rf-depth`: [integer] default: None - Scikit learn 'max_depth' parameter.
	* `--rf-ccp`: [float] default: 0.0 - Scikit learn 'ccp_alpha' parameter. (https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).
	* `--autocutoff`: [int] default: 15 - If no cutoff file is used this sets the default interface cutoff value.


- Flags: 
	* `--pymol`: Output pymol session and image of protein with experimental and predicted interfaces overladed. 
	* `-tv`: Output svg image of a randomly sampled tree (for large datasets this can take up a huge amount of time and space) see https://github.com/parrt/dtreeviz for details.
	* `-xg`: Include the use of gradient boosting regression model.


Output:

- `results.csv`: this file contains the fscore, MCC, Roc AUC and PR AUC for each individual method and model. 

- `roc_model.csv` and pr_model.csv: the TRP and FPR by threshold for each individual method and model, can be used to generate specific ROC or PR graphs.

- `fscore_mcc_by_protein`: the individual fscore and mcc for each protein in the test set. 

- `*.joblib`: the trained models from a generate, test or cv run. Move these into the input directory to be used with 'predict' mode. 

- `pairtest.csv`: Comparison of statistical significance between AUCs.
	- top triangle: difference in pairs of AUCs
	- bottom triangle: log(10) of p-values for the difference in pairs of AUCs.
- `proteins`: Directory containing pymol sessions for each protein in the test set.  
- `cvout`: Directory containing the best parameters for each model used in the final prediction, as well as the individual metrics over each cross validation step. 


---
### Special Thanks To:

<p>Dr. Andras Fiser and Dr. Eduardo J Fajardo for insight and guidance.</p> 

<p>Terence Parr and Prince Grover for use of dtreeviz.</p>


---
### Updates:
Please Consult the CHANGELOG.md for all updates

