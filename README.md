# prediction_of_model_editing

This repository documents the code that was used to test the relationship between the optimal location for editing a language model (gpt2-xl) and the probability that the model gives to the output in the different layers.

# Files 

The file editing_experiment.py performs the edits on all the prompts in the dataset and calculates the success (efficacy, specificity, specificity_plus) of each prompt, averaged over a group of new objects. This file can be run by running run_editing_experiment.bash, with the parameters: from_layer, to_layer. Its output is a scores_<from_layer>_<to_layer>.py file.

The file logit_lens_experiment.py calculates the k labels with the highest probability for each prompt in each layer (if TO_OUTPUT = "labels") or the probabilities that the model assigns to the k labels with the highest probability in the last layer for each prompt (if TO_OUTPUT = " final_probs"). The file can be run by running run_logit_lens.bash, with the parameters: from_layer, to_layer.

Inside the analysis folder are the files used to analyze the results. 

The file analysis_statistic.py is used for general statistical analyzes on Optimal Layer to Update (OLU), and the file analysis_logit_lens.py is used to analyze the relationship between the success of editing in the different layers and the results of logit lens. In both of these files, the signatures of the independent functions appear at the end of the file.

The folder rome_files contains the files of the ROME algorithm (with small changes for this project).

# The output structure

The scores file contains a list d of dictionaries (There is one dictionary in this file for every tuple in neighborhood.py, in order. Each dictionary belongs to a single relation, but there are relations with 2 dictionaries). the structure of each dictionary is:
{'causal_features': dict, 'neighborhood_scores': list, 'efficacy_scores': list, 'paraphrase_scores': list}.

The causal_features dict contains features of the causal trace pattern of each prompt. This dictionary, and the paraphrase_scores list, are not used in this study.

For neighborhood_scores list:
- The neighborhood_scores list contains 48 lists - one list for each layer in the model.
- Each layer-list contains 5 lists - one list for each type of specificity: final (percentage of final outputs that did not change); probs to new (the change in the probability of the new output, normalized); prob to true (the complement of the change in the probability of the true output, normalized); specificity-plus final (here it is empty, because of division by 0 problems); specificity-plus prob to true.
- Each specificuty-type list contains N_subjects floats - score for every subject, by their order in  neighborhood.py.

Similarly, the efficacy_scores has size of (48 X 2 X N_subjects). The 2 types of efficacy are: by final, by prob.






