# prediction_of_model_editing

This repository documents the code that was used to test the relationship between the optimal location for editing a language model (gpt2-xl) and the probability that the model gives to the output in the different layers.

# Files 

The file editing_experiment.py performs the edits on all the prompts in the dataset and calculates the success (efficacy, specificity, specificity_plus) of each prompt, averaged over a group of new objects. This file can be run by running run_editing_experiment.bash, with the parameters: from_layer, to_layer. Its output is a scores_<from_layer>_<to_layer>.py file.

The file logit_lens_experiment.py calculates the k labels with the highest probability for each prompt in each layer (if TO_OUTPUT = "labels") or the probabilities that the model assigns to the k labels with the highest probability in the last layer for each prompt (if TO_OUTPUT = " final_probs"). The file can be run by running run_logit_lens.bash, with the parameters: from_layer, to_layer.

Inside the analysis folder are the files used to analyze the results. 

The file analysis_statistic.py is used for general statistical analyzes on Optimal Layer to Update (OLU), and the file analysis_logit_lens.py is used to analyze the relationship between the success of editing in the different layers and the results of logit lens. In both of these files, the signatures of the independent functions appear at the end of the file.

# The output structure

The scores file contains a list d of dictionaries (There is one dictionary in this file for every dictionary in the neighborhood.py file, in order. Each dictionary belongs to a single relation, but there are relations with 2 dictionaries). the structure of each dictionary is:
{'causal_features': dict, 'neighborhood_scores': list, 'efficacy_scores': list, 'paraphrase_scores': list}.

The causal_features dict contains features of the causal trace pattern of each prompt. This dictionary, and the paraphrase_scores list, are not used in this study.

The neighborhood_scores list contains 48 lists - one list for each layer in the model.

Each layer-list contains 5 lists - one list for each type of specificity: final (percentage of final outputs that did not change); probs (the complement of the change in the probability of the correct output, normalized); 





