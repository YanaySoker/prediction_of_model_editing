from transformers import GPT2Tokenizer, GPT2Model, TFGPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from corrected_neighborhood import d
import sys

# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-j-6B')
# model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B').to("cuda")
model_name = "gpt2-xl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

K = 5


def combine_prompt(subject, relation):
  if relation is not None:
    pref, suff = relation.split("{}")
    prompt = f"{pref}{subject}{suff}"
  else:
    prompt = subject
  return prompt


def k_most_likely_one_layer(probs, k, return_probs = False):
  labels = []
  max_probs = []

  for _ in range(k):
      
    prob, preds = torch.max(probs, dim=0)
    probs[preds] = 0
    label = tokenizer.decode(preds)[1:]
    prob = prob.item()
    labels.append(label)
    max_probs.append(prob)
  
  if return_probs:
        return labels, max_probs
  return labels


def print_dict(dict, file_name=None):
  func = print
  if file_name:
    file = open(file_name, "w", encoding="utf-8")
    func = file.write

  func("d = {\n")
  for key in dict.keys():
    func(f"\t\"{key}\": {dict[key]},\n")
  func("}")

  if file_name:
    file.close()


print("Start")
start_line, end_line = int(sys.argv[1]), int(sys.argv[2])

res_dict = {}

for neighborhood in d[start_line:end_line]:
  relation, subjects, _, _ = neighborhood
  if relation not in res_dict.keys():
    res_dict[relation] = {}
  
  for subject in subjects: 
    text = combine_prompt(subject, relation)
    encoded_input = tokenizer(text, return_tensors='pt').to('cuda')
    output = model(**encoded_input, return_dict=True, output_hidden_states = True, output_hidden_states_mlp = True)

    hidden_states_mlp = output.hidden_states
#     hidden_states = output.hidden_states
    labels_list = []
    print("subject:", subject)
    
    # For FLS:
    
#     for l in range(len(hidden_states)):
#       logits = model.lm_head(hidden_states[l])
#       probs = torch.softmax(logits[:, -1], dim=1)
#       labels = k_most_likely_one_layer(probs[0], K)
#       labels_list.append(labels)    
    
    # For probs:
    
    labels_of_all_layers = []
    max_probs_of_all_layers = []
    for l in range(len(hidden_states_mlp)):   # new: mlp
        logits = model.lm_head(hidden_states_mlp[l])   # new: mlp
        probs = torch.softmax(logits[:, -1], dim=1)
        labels, max_probs = k_most_likely_one_layer(probs[0], K, return_probs=True)
        
        labels_of_all_layers.append(labels)
        max_probs_of_all_layers.append(max_probs)

    res_dict[relation][subject] = (labels_of_all_layers, max_probs_of_all_layers)
  
  print("DONE relation:", relation)

# For FLS:
# print_dict(res_dict, f"labels_dict_gpt_j_{start_line}_{end_line}.py")

print_dict(res_dict, f"probs_in_last_layer_{start_line}_{end_line}.py")
# torch.save(res_dict, f"probs_dict_{start_line}_{end_line}.pt")