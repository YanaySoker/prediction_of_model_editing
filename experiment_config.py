from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rome_files.experiments.causal_trace import ModelAndTokenizer
from rome_files.dsets import KnownsDataset
from rome_files.experiments.causal_trace import collect_embedding_std
from rome_files.util.globals import DATA_DIR

# model_name = "EleutherAI/gpt-j-6B" 
# model_name = "gpt-j-6B"
model_name = "gpt2-xl"


# mt = ModelAndTokenizer(
#     model_name,
#     torch_dtype=(torch.float16 if "20b" in model_name else None),
# )
# tok = AutoTokenizer.from_pretrained(model_name)

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

class MT:
    pass

mt = MT()
mt.model = model
mt.tokenizer = tok

tok.pad_token = tok.eos_token

M = dict()

if model_name == "gpt2-xl":
    NUM_OF_LAYERS = 48
    LAYERS_RANGE = range(0, NUM_OF_LAYERS,5)
    HIGH_RES_LAYERS_RANGE = range(0, NUM_OF_LAYERS)
else: 
    NUM_OF_LAYERS = 28
    LAYERS_RANGE = range(0, NUM_OF_LAYERS,3)
    HIGH_RES_LAYERS_RANGE = range(0, NUM_OF_LAYERS)

knowns = KnownsDataset(DATA_DIR)
noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
prefix = "./rome/"

counterfacts_url = "https://rome.baulab.info/data/dsets/counterfact.json"
