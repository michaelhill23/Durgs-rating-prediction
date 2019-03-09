# Durgs-rating-prediction
Classification problem for predicting drugs rating based on text reviews.<br>
Holistic research description of the problem can be found in Jupyter notebook file. **0_assignment.ipynb**
Please start with **0_assignment.ipynb** while reviewing research solution.
## DESCRIPTIONS OF INCLUDED FILES:
**0_assignment.ipynb** - file with final description of the solution<br><br>
**1_cleaning_data.py** - Python code for cleanig initial test and train data sets and creating 'cleaned' test and train data sets for further processing.<br> 
INPUT FILES:<br> 
*training_data.tsv<br>
*test_data.tsv<br>
OUTPUT FILES:<br>
*train_output.csv*<br>
*test_output.csv*<br><br>
**2_feature_selection.py** - Python code for feature engeeniring and choosing features for further processing<br><br>
**4_parameters_tunning.py** - Python code for tunning the parameters of selected classification models<br> <br>
**5_scoring_tunned_parameters.py** - Python code for reviewing scores of selected models and choosing the final model for the solution<br><br>
**6_files_for_final_plots.py** - Python code for creating files that will be used to present prediction results in 0_assignment.ipynb<br>
OUTPUT FILES:<br>
*conditions.csv*<br>
*drugs.csv*<br><br>
**7_final_plots.py** - Python code for creating plots that present prediction results<br><br>
**training_data.tsv** - Initial training data set file with 'dirty' patients' drugs reviews. It is an input for code in 1_cleaning_data.py<br><br>
**test_data.tsv** - Initial test data set file with 'dirty' patients' drugs reviews. It is an input for code in 1_cleaning_data.py<br><br>
**train_output.csv** - File with 'cleaned' training data set. Consists of cleaned drug reviews and additional features added after reviews' sentiment analysis<br><br>
**test_output.csv** - File with 'cleaned' test data set. Consists of cleaned drug reviews and additional features added after reviews' sentiment analysis<br><br>
**conditions.csv** - File consisting of drug rating prediction results grouped by patients' conditions. It is used for final presentation of results in 0_assignment.ipynb file<br><br>
**drugs.csv** File consisting of drug rating prediction results grouped by drug name. It is used for final presentation of results in 0_assignment.ipynb file<br><br>
