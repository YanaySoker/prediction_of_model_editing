from transformers import GPT2Tokenizer, GPT2Model, TFGPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from neighborhood import d
import sys

# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-j-6B')
# model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B').to("cuda")
model_name = "gpt2-xl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

K = 5
TO_OUTPUT = "labels"   # "labels", "final_probs"


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
    output = model(**encoded_input, return_dict=True, output_hidden_states = True)

#     hidden_states_mlp = output.hidden_states
    hidden_states = output.hidden_states
    print("subject:", subject)
    
    if TO_OUTPUT=="labels":
      res_s_list = []
      for l in range(len(hidden_states)):
        logits = model.lm_head(hidden_states[l])
        probs = torch.softmax(logits[:, -1], dim=1)
        labels = k_most_likely_one_layer(probs[0], K)
        res_s_list.append(labels)    
 
    if TO_OUTPUT=="final_probs":
      logits = model.lm_head(hidden_states[-1])
      probs = torch.softmax(logits[:, -1], dim=1)
      labels, max_probs = k_most_likely_one_layer(probs[0], K, return_probs=True)
      res_s_list = (labels, max_probs)

    res_dict[relation][subject] = res_s_list
  
  print("DONE relation:", relation)

dict_name = "probs_in_last_layer" if TO_OUTPUT=="final_probs" else "labels_dict"
print_dict(res_dict, f"{dict_name}_{start_line}_{end_line}.py")
