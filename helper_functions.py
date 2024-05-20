from util import nethook
import gc
import torch, numpy
from experiments.causal_trace import predict_from_input
from experiment_config import *


def combine_prompt(subject, relation):
  if relation is not None:
    pref, suff = relation.split("{}")
    prompt = f"{pref}{subject}{suff}"
  else:
    prompt = subject
  return prompt


def is_in_prompt(relation, prompt):
  pref, suff = relation.split("{}")
  return pref==prompt[:len(pref)] and suff==prompt[-len(suff):]


def get_subject(prompt, relation):
  pref, suff = relation.split("{}")
  start = len(pref)
  end = len(prompt)-len(suff)
  subject = prompt[start:end]
  return subject


def sum_matrices(A, B, alpha):
  # adding alpha*B to A:
  for i in range(len(A)):
    row = A[i]
    for j in range(len(row)):
      if not isinstance(B[i][j], EMPTY):
        to_add = alpha * B[i][j] 
        row[j]+=to_add
        

def stack_matrices(A, B):
  # A (B,n,m), B (B,n) --> (B, n, m+1)
  if len(A)==0:
      for b in B:
          new_row = []
          for b_ in b:
              new_row.append([b_])
          A.append(new_row)
      return

  for i in range(len(A)):
    row = A[i]
    for j in range(len(B[i])):
      if not isinstance(B[i][j], EMPTY):
        row[j].append(B[i][j])
      else:
        row[j].append("*")


def predict_from_input_with_probs(model, inp):    # based on origin "predict_from_input"
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)   
    p, preds = torch.max(probs, dim=1)        
    return preds, p, probs[0]


## predict_token
def predict_all_from_input(model, inp):    # based on origin "predict_from_input"
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    return probs


def predict_token(mt, prompts, return_p=False, return_idx = False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    elif return_idx:
        preds = preds[0]
        result = (result, preds)

    return result


def predict_by_idx(mt, prompt, idx):     # yanay
  # model, str, int --> float
  # idx: index of object we want to know its probability
  inp = make_inputs(mt.tokenizer, [prompt])
  preds = predict_all_from_input(mt.model, inp)
  return preds[0][idx].item()


def predict_by_idx_and_max(mt, prompt, idxs, return_idx_and_p=False):     # yanay
  # idxs = list of objects index 
  inp = make_inputs(mt.tokenizer, [prompt])
  preds, p, probs = predict_from_input_with_probs(mt.model, inp)
  relevant_probs = [probs[idx].item() for idx in idxs]

  token = [mt.tokenizer.decode(c) for c in preds]
  
  if return_idx_and_p:
    pred_token = (token[0][1:], preds.item(), p.item())
  else:
    pred_token = token[0][1:]
  
  return pred_token, relevant_probs


def naiv_predict(subject, relation = None, return_idx = False):
  prompt = combine_prompt(subject, relation)

  t = predict_token(
    mt,
    [prompt],
    return_p=False,
    return_idx = return_idx
  )
  
  if return_idx:
    return t[0][0][1:], t[1]
  return t[0][1:]


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


def print_list(list_input, file_name=None):
  func = print
  if file_name:
    file = open(file_name, "w", encoding="utf-8")
    func = file.write

  func("d = [\n")
  for item in list_input:
    func(f"\t{item},\n")
  func("]")

  if file_name:
    file.close()


def clean(mt):
    print(clean)
    if "orig_weights" in M.keys():
        with torch.no_grad():
            for k, v in M["orig_weights"].items():
                nethook.get_parameter(mt.model, k)[...] = v
        print("Original model restored")
    else:
        print(f"No model weights to restore")
    
    torch.cuda.empty_cache()
    gc.collect()
    if M is not None:
        M.clear()


def encode(token):
  if token[0]!=" ":
    token = " "+token
    return mt.tokenizer.encode(token)[0]


class EMPTY():
   pass

