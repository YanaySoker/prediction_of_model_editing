
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution, LAYER_IDX
from paraphrase_dictionary import d as paraphrase_dict
from generation_dictionary import d as generation_dict

import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    # predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset

from helper_functions import *
from experiment_config import *

torch.set_grad_enabled(True)

import random
_seed = 1
random.seed(_seed)
numpy.random.seed(seed=_seed)
torch.manual_seed(_seed)

import copy

def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    print("result:\n",result)
    plot_trace_heatmap(result, savepdf, modelname=modelname)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None):
    for kind in ["mlp"]:
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind
        )


def neighbors_probs(mt, main_subject, relation, neighbors, target_id):
  # str, list(str), int --> list(float)
  probs = []
  for neighbor in neighbors:
    if main_subject is None or neighbor!=main_subject:
      prompt = combine_prompt(neighbor, relation)
      probs.append(predict_by_idx(mt, prompt, target_id))
  return probs

def neighbors_true_probs(mt, main_subject, relation, neighbors, target_ids):
  # str, list(str), int --> list(float)
  probs = []
  for i in range(len(neighbors)):
    neighbor = neighbors[i]
    if main_subject is None or neighbor!=main_subject:
      prompt = combine_prompt(neighbor, relation)
      probs.append(predict_by_idx(mt, prompt, target_ids[i]))
  return probs

def neighbors_probs_from_orig(mt, main_subject, relation, neighbors, targets_id):
  # str, list(str), int --> list(float)
  probs = []
  for i in range(len(neighbors)):
    neighbor=neighbors[i]
    if main_subject is None or neighbor!=main_subject:
      prompt = combine_prompt(neighbor, relation)
      probs.append(predict_by_idx(mt, prompt, targets_id[i]))
  return probs

def neighboring(probs1, probs2):
  # list(float), list(float) --> float
  # tensor(float), tensor(float), --> float

  m = len(probs1)
  numerator = torch.abs(probs1-probs2)
  denominator = torch.abs(probs1-0.5) + 0.5
  ngbring_vector = 1 - numerator / denominator
  ngbring = ngbring_vector.sum() / m
  return ngbring.item()

  # f = []
  # for i in range(m):
  #   numerator = abs(probs1[i]-probs2[i])
  #   denominator = 0.5+abs(probs1[i]-0.5)
  #   ngbring = 1 - numerator / denominator
  #   f.append(ngbring)
  # return sum(f) / m

def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None):
    for kind in ["mlp"]:
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind
        )


FILE_NAME_SANITY = "another_test"
def change_and_check_sanity(subject, relation, new_target, layer_idx):
  clean(mt)
  LAYER_IDX[0] = layer_idx

  random.seed(_seed)
  numpy.random.seed(seed=_seed)
  torch.manual_seed(_seed)
  
  pre_output = naiv_predict(subject, relation)

  request = [
      {
          "prompt": relation,
          "subject": subject,
          "target_new": {"str": new_target},
      }
  ]

  M["model_new"], M["orig_weights"] = demo_model_editing(mt.model, tok, request, ["a"], alg_name="ROME")
  mt.model = M["model_new"]

  post_output = naiv_predict(subject, relation)
  file = open(FILE_NAME_SANITY, "a", encoding="utf-8")
  file.write(f"relation = {relation}, \tsubject = {subject}, \tnew target = {new_target}, \tlayer = {layer_idx}, \toriginal output = {pre_output}, \tnew output = {post_output}, \tsuccess = {post_output.replace(' ', '') == new_target.replace(' ', '')}\n")
  file.close()

    
def change_and_check(main_subject_idx, relation, target_for_new, neighborhood, orig_probs_to_new, orig_and_prob, to_target_flag, preplus, pre_change_target=None):
  # orig_and_prob = {"orig_final": [...], "orig_id_final": [...], "orig_probs_true": [...]}
  # orig_probs_to_new used for: calculate list of the probability for new_target, for each neighbor. 

  # if to_target_flag: count only changes to new_target. else: all change.
  # neighborhood: list(subjects). including main_subject

  clean(mt)

  main_subject = neighborhood[main_subject_idx]

  random.seed(_seed)
  numpy.random.seed(seed=_seed)
  torch.manual_seed(_seed)

  new_target = orig_and_prob["orig_final"][main_subject_idx] if target_for_new=="self" else target_for_new
  try:
    tok_id = encode(new_target)
  except:
    print("\n\n$$$$$$$$\n%%%%%%%%\n")
    print("main_subject_idx:", main_subject_idx)
    print("relation:", relation)
    print("target_for_new:", target_for_new)
    print(f">orig_and_prob['orig_final'][main_subject_idx]<: >{orig_and_prob['orig_final'][main_subject_idx]}<")
    print("\norig_and_prob['orig_final']:\n", orig_and_prob["orig_final"])
    print("\n\n%%%%%%%%\n$$$$$$$$\n")
    return [-1] * 11
    

# pre edit
  if pre_change_target is not None:
    pre_change_target = orig_and_prob["orig_final"][main_subject_idx] if pre_change_target=="self" else pre_change_target 
    request = [
        {
            "prompt": relation,
            "subject": main_subject,
            "target_new": {"str": pre_change_target},
        }
    ]
    M["model_new"], M["orig_weights"] = demo_model_editing(mt.model, tok, request, ["a"], alg_name="ROME")
 
  # edit
  request = [
      {
          "prompt": relation,
          "subject": main_subject,
          "target_new": {"str": new_target},
      }
  ]

  M["model_new"], orig_weights = demo_model_editing(mt.model, tok, request, ["a"], alg_name="ROME")
  if pre_change_target is None:
    M["orig_weights"] = orig_weights
    
  mt.model = M["model_new"]
  results = []

  post_probs_to_target = []
  post_probs_to_true = []
  
  # check: neighborhood
  count=0
  for i in range(len(neighborhood)):
    if i!=main_subject_idx:
      neighbor_prompt = combine_prompt(neighborhood[i], relation)
      true_target_id = orig_and_prob["orig_id_final"][i]
      idxs = [tok_id, true_target_id]
      
      pred_token, relevant_probs = predict_by_idx_and_max(mt, neighbor_prompt, idxs)

      post_probs_to_target.append(relevant_probs[0])
      post_probs_to_true.append(relevant_probs[1])
      
      if to_target_flag:
        count+= 1*(pred_token==new_target)
      else:
        count+= 1*(pred_token!=orig_and_prob["orig_final"][i])
  
  # by finals
  results.append(count / (len(neighborhood)-1))
  
  # by prob to new
  if target_for_new=="self":
        results.append(-1)
  else:
        filtered_orig_probs = orig_probs_to_new[:main_subject_idx]+orig_probs_to_new[main_subject_idx+1:]
        neighboring_score = neighboring(torch.tensor(filtered_orig_probs), torch.tensor(post_probs_to_target))
        results.append(neighboring_score)
  
  # by prob to true
  orig_probs_true = orig_and_prob["orig_probs_true"]
  filtered_orig_probs_true = orig_probs_true[:main_subject_idx]+orig_probs_true[main_subject_idx+1:]
  neighboring_true_score = neighboring(torch.tensor(filtered_orig_probs_true), torch.tensor(post_probs_to_true))
  results.append(neighboring_true_score)

  # check: neighborhood-plus
  post_finalsplus = []
  post_probstrue = []
  for neighbor_idx in range(len(neighborhood)):
      if neighbor_idx!=main_subject_idx:
        neigh = neighborhood[neighbor_idx]
        neigh_prompt = combine_prompt(neigh, relation)
        plus_prompt = main_subject + ". " + neigh_prompt
        trueplus_obj, trueplus_prob = predict_by_idx_and_max(mt, plus_prompt, [orig_and_prob["orig_id_final"][neighbor_idx]])
        post_finalplus = int(trueplus_obj==orig_and_prob["orig_final"][neighbor_idx])
        post_finalsplus.append(post_finalplus)
        post_probstrue.append(trueplus_prob[0])
  
  post_finalplus_score = 1-(preplus["finals score"]-torch.tensor(post_finalsplus)).sum() / preplus["finals score"].sum()
  post_finalplus_score = post_finalplus_score.item()
  results.append(post_finalplus_score)

  post_probstrue_score = neighboring(preplus["prob score"], torch.tensor(post_probstrue))
  results.append(post_probstrue_score)

  # check efficacy
  prompt = combine_prompt(main_subject, relation)
  post_final, post_prob_to_new = predict_by_idx_and_max(mt, prompt, [tok_id])
  
  if target_for_new=="self":
        orig_p_to_new = orig_and_prob["orig_probs_true"][main_subject_idx]
  else:
        orig_p_to_new = orig_probs_to_new[main_subject_idx]
  efficacy_final = int(post_final==new_target)
  results.append(efficacy_final)
  efficacy_prob = (post_prob_to_new[0] - orig_p_to_new) / (1 - orig_p_to_new)
  results.append(efficacy_prob)

#   # check paraphrase
#   if prompt in paraphrase_dict.keys() and target_for_new!="self":
#     paraphrase_score_final = 0
#     paraphrase_score_prob = 0
#     for paraphrase in paraphrase_dict[prompt]:
#       post_final_par, post_prob_to_new_par = predict_by_idx_and_max(mt, paraphrase, [tok_id])
#       pre_prob_to_new_par = paraphrase_and_generation_dict["paraphrase"][paraphrase]["new_objects_p"][tok_id]
#       final_score_par = int(post_final_par==new_target)
#       prob_score_par = (post_prob_to_new_par[0] - pre_prob_to_new_par) / (1 - pre_prob_to_new_par)
#       paraphrase_score_final += final_score_par
#       paraphrase_score_prob += prob_score_par
#     results.append(paraphrase_score_final / len(paraphrase_dict[prompt]))
#     results.append(paraphrase_score_prob / len(paraphrase_dict[prompt]))
#   else:
#     results.append(EMPTY())
#     results.append(EMPTY())
  
#   # check generation
#   if prompt in generation_dict.keys() and target_for_new!="self":
#     generation_score_final = 0
#     generation_score_prob = 0
#     for generation in generation_dict[prompt]:
#       post_final_gen, post_prob_to_new_gen = predict_by_idx_and_max(mt, generation, [tok_id])
#       pre_prob_to_new_gen = paraphrase_and_generation_dict["generation"][generation]["new_objects_p"][tok_id]
#       final_score_gen = int(post_final_gen==new_target)
#       prob_score_gen = (post_prob_to_new_gen[0] - pre_prob_to_new_gen) / (1 - pre_prob_to_new_gen)
#       generation_score_final += final_score_gen
#       generation_score_prob += prob_score_gen
#     results.append(generation_score_final / len(generation_dict[prompt]))
#     results.append(generation_score_prob / len(generation_dict[prompt]))
#   else:
#     results.append(EMPTY())
#     results.append(EMPTY())

  # [single float, ... , single float]   --> len = 7 
  return results

def neighborhood_score_by_object(neighborhood_data, target, orig_probs_new, orig_and_prob, to_target_flag, pre_change_target):
  # Calculate lists (one or two) of neighborhood_score of specific target (object) (given specific relation) over all subjects.
  # neighborhood_data: (relation: str, subjects: list(str), orig_objects: list(str), new_objects: list(str))

  relation, subjects, _, _ = neighborhood_data

  neighborhood_scores = []
  neighborhood_scores.append([])
  neighborhood_scores.append([])
  neighborhood_scores.append([])
  neighborhood_scores.append([])
  neighborhood_scores.append([])

  efficacy_scores = []
  efficacy_scores.append([])
  efficacy_scores.append([])

  # orig_and_prob = {"orig_final": [], "orig_id_final": [], "orig_probs_true": []}
  # pred_token, relevant_probs = predict_by_idx_and_max(mt, prompt, new_objects_id, return_idx_and_p=True)
  # true_obj, true_obj_id, p = pred_token
  
  for subject_idx in range(len(subjects)):
    subj = subjects[subject_idx]
    pre_finalsplus = []     # for each neighbor: whether the output is correct for the prompt plus the (unrelated) subject at the beginning (boolean)
    pre_probstrue = []      # soft pre_finalsplus: the probs for the correct output for the prompt plus the subject
    for neighbor_idx in range(len(subjects)):
      if neighbor_idx!=subject_idx:
        neigh = subjects[neighbor_idx]
        neigh_prompt = combine_prompt(neigh, relation)
        plus_prompt = subj + ". " + neigh_prompt
        trueplus_obj, trueplus_prob = predict_by_idx_and_max(mt, plus_prompt, [orig_and_prob["orig_id_final"][neighbor_idx]])
        pre_finalplus = int(trueplus_obj==orig_and_prob["orig_final"][neighbor_idx])

        
        pre_finalsplus.append(pre_finalplus)
        pre_probstrue.append(trueplus_prob[0])
        
    preplus = {"finals score": torch.tensor(pre_finalsplus), "prob score": torch.tensor(pre_probstrue)}
    
#     actual_target = orig_and_prob["orig_final"][subject_idx] if target=="self" else target
    
    if target=="self":
        orig_probs = None
    else:
        orig_probs = orig_probs_new[target]
    
    current_scores = change_and_check(subject_idx, relation, target, subjects, orig_probs, orig_and_prob, to_target_flag, preplus, pre_change_target)

    neighborhood_scores[0].append(current_scores[0])
    neighborhood_scores[1].append(current_scores[1])
    neighborhood_scores[2].append(current_scores[2])
    neighborhood_scores[3].append(current_scores[3])
    neighborhood_scores[4].append(current_scores[4])

    efficacy_scores[0].append(current_scores[5])
    efficacy_scores[1].append(current_scores[6])

  
  # neighborhood_scores: [[floats] ... [floats]]   --> len = 5
  # efficacy_scores: [[floats], [floats]]          --> len = 2
  # len of each [floats] is: |subjects| 
  return neighborhood_scores, efficacy_scores


def neighboring_score_of_neighborhood(neighborhood_data, pre_outputs, to_target_flag=True, experiment_type="singel"):

  _, subjects, _, new_objects = neighborhood_data
  n_subjects = len(subjects)
  n_objects = len(new_objects)
  
  neighborhood_scores = []
  neighborhood_scores.append([0]*n_subjects)   # scores by final
  neighborhood_scores.append([0]*n_subjects)   # scores by to target probs
  neighborhood_scores.append([0]*n_subjects)   # scores by true object probs
  neighborhood_scores.append([0]*n_subjects)   # plus by final
  neighborhood_scores.append([0]*n_subjects)   # plus by true object probs

  efficacy_scores = []
  efficacy_scores.append([0]*n_subjects)   # scores by final
  efficacy_scores.append([0]*n_subjects)   # scores by to target probs 

  for obj_i in range(len(new_objects)): 
    target = new_objects[obj_i]
        
    if experiment_type=="serial":
        pre_change_target = new_objects[obj_i-1]
    if experiment_type=="reverse":
        pre_change_target = target
        target = "self"
    if experiment_type=="single":
        pre_change_target = None
    
    orig_probs_new = pre_outputs[1]
#     orig_probs = pre_outputs[1][target]
    orig_and_prob = pre_outputs[0]
    current_neighborhood_scores, current_efficacy_scores = neighborhood_score_by_object(neighborhood_data, target, orig_probs_new, orig_and_prob, to_target_flag, pre_change_target)
    
    sum_matrices(neighborhood_scores, current_neighborhood_scores, alpha=1/n_objects)
    sum_matrices(efficacy_scores, current_efficacy_scores, alpha=1/n_objects)

  # [[floats], [floats], ... [floats]], [[floats], [floats]]
  #  5 times                            2 times               
  return neighborhood_scores, efficacy_scores


def neighborhood_results(neighborhood_data, to_target_flag=False, num_of_layers=28, experiment_type="single", layers_list=LAYERS_RANGE):
#   causal_features = {"max layers": [], "entropies": [], "maxs": [], "mins": [], "avrgs": [], "effs": []}
  results = {"neighborhood_scores": [], "efficacy_scores": []}
  relation, subjects, _, new_objects = neighborhood_data

  orig_and_prob = {"orig_final": [], "orig_id_final": [], "orig_probs_true": []}
  
  orig_probs_new = {new_object: [] for new_object in new_objects}

  pre_outputs = (orig_and_prob, orig_probs_new)   # first dict: original output
  
  new_objects_id = []
  for new_object in new_objects:
    new_objects_id.append(encode(new_object))
  
#   prompts_with_paraphrase = paraphrase_dict.keys()
#   prompts_with_generation = generation_dict.keys()
#   paraphrase_and_generation_dict = {"paraphrase": dict(), "generation": dict()}
  to_delete = []
  
  for subject in subjects:
    try:        
      prompt = combine_prompt(subject, relation)
    
#       max_layer, entropy, _, _max, _min, avrg, effs = max_layer_and_entropy(prompt, subject, max_neighbors=[], effect_idx=range(num_of_layers))
    
#       causal_features["max layers"].append(max_layer)
#       causal_features["entropies"].append(entropy)
#       causal_features["maxs"].append(_max)
#       causal_features["mins"].append(_min)
#       causal_features["avrgs"].append(avrg)
#       causal_features["effs"].append(effs)

      pred_token, relevant_probs = predict_by_idx_and_max(mt, prompt, new_objects_id, return_idx_and_p=True)

      true_obj, true_obj_id, p = pred_token
      pre_outputs[0]["orig_final"].append(true_obj)
      pre_outputs[0]["orig_id_final"].append(true_obj_id)
      pre_outputs[0]["orig_probs_true"].append(p)

      for i in range(len(new_objects)):
        pre_outputs[1][new_objects[i]].append(relevant_probs[i])
        
#       if prompt in prompts_with_paraphrase:
#         for paraphrase in paraphrase_dict[prompt]:
#           try:
#             new_dict = dict()
#             pred_token, relevant_probs = predict_by_idx_and_max(mt, prompt, new_objects_id, return_idx_and_p=True)
#             new_dict["orig_p"] = pred_token[2]
#             relevant_probs_dict = {new_objects_id[k]: relevant_probs[k] for k in range(len(relevant_probs))}
#             new_dict["new_objects_p"] = relevant_probs_dict
#             paraphrase_and_generation_dict["paraphrase"][paraphrase] = new_dict
#           except:
#             print(f"\n\n###############\n%%%%%%%%%%%%%%%\ndelete paraphrase:\n{paraphrase}\n%%%%%%%%%%%%%%%\n###############\n\n")
      
#       if prompt in prompts_with_generation:
#         for generation in generation_dict[prompt]:
#           try:
#             new_dict = dict()
#             pred_token, relevant_probs = predict_by_idx_and_max(mt, prompt, new_objects_id, return_idx_and_p=True)
#             new_dict["orig_p"] = pred_token[2]
#             relevant_probs_dict = {new_objects_id[k]: relevant_probs[k] for k in range(len(relevant_probs))}
#             new_dict["new_objects_p"] = relevant_probs_dict
#             paraphrase_and_generation_dict["generation"][generation] = new_dict
#           except:
#             print(f"\n\n###############\n%%%%%%%%%%%%%%%\ndelete generation:\n{generation}\n%%%%%%%%%%%%%%%\n###############\n\n")
    except:
      print(f"\n\n###############\n%%%%%%%%%%%%%%%\ndelete:\nrelation = {relation}\nsbject = {subject}\n%%%%%%%%%%%%%%%\n###############\n\n")
      to_delete.append(subject)
  
  for subj_to_delete in to_delete:
    subjects.remove(subj_to_delete)
    
  for layer_idx in layers_list:
    LAYER_IDX[0] = layer_idx
    neighboring, efficacy_scores = neighboring_score_of_neighborhood(neighborhood_data, pre_outputs, to_target_flag, experiment_type)
    
    results["neighborhood_scores"].append(neighboring)
    results["efficacy_scores"].append(efficacy_scores)
  
  # results {one neighborhood} = {"causal_features": {<feature>: [numbers...]}, "neighborhood_scores": [[[floats], [floats]... [floats]], [[floats], [floats]... [floats]], ...]}
  #                                                               len(subjects)                        len(layers), 7, len(subjects)
  return results 

import transformers
def all_results(neighborhood_list, to_target_flag=False, num_of_layers=NUM_OF_LAYERS, experiment_type="single", HIGH_RES_FLAG = False):
  # orig_probs_flag: to compute neighboring according to the probabilities 
  # orig_final_flag: to compute neighboring according to the final output
  # to_target_flag: for final output - count only the changes towards the new object.
  results = []
  print("start")
  for t in range(len(neighborhood_list)):
    neighborhood_data = neighborhood_list[t]
    
    for s in neighborhood_data[1]:
        neighbor_prompt = combine_prompt(s, neighborhood_data[0])
        pred_token, _ = predict_by_idx_and_max(mt, neighbor_prompt, [0])
        print(neighbor_prompt, "-->", pred_token)
    
    print("\tdata idx =", t, "\t")
    clean(mt)
    if HIGH_RES_FLAG:
        _layers_list = HIGH_RES_LAYERS_RANGE
    else:
        _layers_list = LAYERS_RANGE
    new_results = neighborhood_results(neighborhood_data, to_target_flag, num_of_layers, experiment_type, layers_list=_layers_list)
    results.append(new_results)
  return results
