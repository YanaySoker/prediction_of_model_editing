import torch
from util import nethook
import gc


from transformers import AutoModelForCausalLM, AutoTokenizer
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

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
from config import *

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


def predict_from_input_with_probs(model, inp):    # yanay; based on origin "predict_from_input"
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)   
    p, preds = torch.max(probs, dim=1)        
    return preds, p, probs[0]


## predict_token
def predict_all_from_input(model, inp):    # yanay
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


def subset(sorted_list):
  # count_list.sort()
  S = sum(sorted_list)
  start, end = 0, 0
  while sum(sorted_list[start:end])<S/2:
    end+=1
  while sum(sorted_list[start:end])>S/2:
    start+=1
  if start==end:
    return range(end-1)
  return range(start,end)


import math

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState()  ### For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )

def return_map(
    prompt,
    subject,
    mt=mt,
    noise=noise_level,
    samples=10,
    window=10,
    kind="mlp",
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    return result


def generate_city_prompt(city):
  prompt = "is the capital city of"
  word=naiv_predict(city, prompt)
  while word in ["the", "state", "State", "of", "Republic", "province", "Province"]:
    prompt = prompt+ " " + word
    word=naiv_predict(city, prompt)
  return prompt


def entropy(tens):
    tens_norm = tens / tens.sum()
    logs = torch.log2(tens_norm)
    logs = torch.where(logs==-float("inf"),0,logs)
    y = logs * tens_norm
    return -y.sum().item() / math.log2(len(tens))


def max_layer_and_entropy(prompt, subject, max_neighbors=[1], effect_idx=[]):
  result = return_map(prompt, subject)
  scores = result['scores']
  a, b = result['subject_range']
  argmax = scores[a:b].argmax().item()

  relevant_token_idx = int(argmax / len(scores[0])) + a
  relevant_token = scores[relevant_token_idx]

  _max = scores[a:b].max().item()
  _min = scores[a:b].min().item()
  avrg = relevant_token.sum().item() / (len(scores[0]))
  effs = []
  for idx in effect_idx: 
    effs.append(relevant_token[idx].item())

  layer = argmax % len(scores[0])
  cent = []
  for i in max_neighbors:
    if layer+i>=0:
      cent.append(((relevant_token[layer] - relevant_token[layer+1]) / relevant_token[layer]).item())
    else:
      cent.append(-1)
  return layer, entropy(relevant_token), cent, _max, _min, avrg, effs


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

