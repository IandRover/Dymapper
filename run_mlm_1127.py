import os 
import logging
logging.disable(logging.INFO) 

import torch
import torch.nn as nn
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    Trainer,
    is_torch_tpu_available,
)
from transformers.utils.versions import require_version
import ipdb, copy

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from utils_mlm.parser import get_HF_Args
parsera = get_HF_Args()
model_args, data_args, training_args = parsera.parse_json_file("./configs/mlm_wikitext.json")

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--run_name', type=str, default="mlm_1122")
parser.add_argument('--from_scratch', type=int, default=0)
parser.add_argument('--attn_var', type=str, default="")
parser.add_argument('--attn_map', type=str, default="")
parser.add_argument('--n_blocks', type=int, default=12)
parser.add_argument('--n_registers', type=int, default=2)
parser.add_argument('--n_heads', type=int, default=12)
parser.add_argument('--head_dim', type=int, default=64)
parser.add_argument('--layer_indices', nargs='+', default=[100], type=int, help='an integer for the accumulator')
parser.add_argument('--seed', default=0, type=int)
args, _ = parser.parse_known_args()

from utils_mlm.misc import misc
train_dataset, eval_dataset, model, data_collator, tokenizer, compute_metrics, preprocess_logits_for_metrics = misc(logging, training_args, data_args, model_args, logger, args)

if data_args.dataset_config_name == "wikitext-103-v1":
    args.run_name += f"_WK103"

if args.from_scratch == 0:
    args.run_name += f"_FT"
else:
    args.run_name += f"_RI"

if args.attn_map != "": args.run_name += f"_{args.attn_map}"
    
if any(element in range(1,13) for element in args.layer_indices):
    args.run_name += f"_L{''.join(str(number) for number in args.layer_indices)}"
elif 99 in args.layer_indices:
    args.run_name += f"_BL"
elif 100 in args.layer_indices:
    if args.n_blocks == 12:
        args.run_name += f"_Lall"
        args.layer_indices = range(1,13)
    elif args.n_blocks in range(1,12):
        args.run_name += f"_B{args.n_blocks}"
    else:
        raise ValueError("Please specify valid number of blocks")
else:
    raise ValueError("Please specify the layer indices")

if 99 in args.layer_indices:
    print(f"Using Original Transformer Attention")
elif args.n_registers in [1,2,3,4,8,16,32]:
    args.run_name += f"_nR{args.n_registers}"
else:
    raise ValueError("Please specify valid number of registers")

if args.n_heads != 12: 
    args.run_name += f"_nH{args.n_heads}"
    args.att_dim = args.n_heads * args.head_dim

if args.attn_var != "": args.run_name += f"_{args.attn_var}"

training_args.run_name = args.run_name
training_args.seed = args.seed
training_args.output_dir = os.path.join(training_args.output_dir, args.run_name)
if "mlm_1123" not in training_args.run_name: assert 0 == 1, f"Please make sure about the name of model"
print(f"Session Name: {training_args.run_name}")
print(f"Output Dir: {training_args.output_dir}")

class RobertaSelfAttention_identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        return (hidden_states, attention_probs) if output_attentions else (hidden_states,)
    

class RobertaSelfAttention_matchKV(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.attn_var = args.attn_var
        self.attn_map = args.attn_map

        self.layer_index = args.layer_index
        self.num_attention_heads = args.n_heads
        self.attention_head_size = args.head_dim
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.K1 = nn.Linear(args.hidden_size, self.all_head_size)
        nn.init.xavier_normal_(self.K1.weight)
        self.ReadingHead = nn.Parameter(torch.zeros((self.num_attention_heads, self.attention_head_size)))
        nn.init.xavier_normal_(self.ReadingHead)
        self.V1 = nn.Linear(args.hidden_size, self.all_head_size)
        nn.init.xavier_normal_(self.V1.weight)
        self.P1 = nn.Linear(self.all_head_size, args.hidden_size)
        nn.init.xavier_normal_(self.P1.weight)
        self.ReadingHead = nn.Parameter(torch.zeros((self.num_attention_heads, self.attention_head_size)))
        nn.init.xavier_normal_(self.ReadingHead)

        if self.attn_var == "Soft1":
            self.bias_h = nn.Parameter(torch.zeros((self.num_attention_heads)))
        elif self.attn_var == "Soft2":
            self.bias_h = nn.Parameter(torch.ones((self.num_attention_heads)))

        # self.P1 = nn.Linear(args.all_head_size, self.hidden_size)
        # self.G1 = nn.Linear(args.hidden_size, self.hidden_size)

        self.act = nn.ReLU(inplace=False)

        # memory mapping 
        print(f"Layer: {self.layer_index}: {args.run_name} {args.attn_var}")
        self.run_name = args.run_name
        self.n_registers = args.n_registers
        self.bidirection_weight = nn.Parameter(torch.ones((1,1,self.num_attention_heads,1,self.n_registers*2))*(1/self.n_registers/2))
        self.sigmoid = nn.Sigmoid()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size()[:-1] + (self.num_attention_heads, self.attention_head_size))

    def _softmax(self, x):
        maxes = torch.max(x, 1, keepdim=True)[0]
        x_exp = torch.exp(x-maxes)
        return x_exp / torch.sum(x_exp, 1, keepdim=True) + 1 / torch.exp(maxes)

    def _softmax2(self, x):
        maxes = torch.max(x, 1, keepdim=True)[0]
        x_exp = torch.exp(x-maxes)
        return x_exp / torch.sum(x_exp, 1, keepdim=True)

    def _softmax3(self, x):
        maxes = torch.max(x, 1, keepdim=True)[0]
        x_exp = torch.exp(x-maxes)
        return x_exp / (torch.sum(x_exp, 1, keepdim=True) + 1 / torch.exp(maxes))

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):

        DEVICE = hidden_states.device

        if self.attn_var == "KV1": 
            K1 = self.transpose_for_scores(self.act(self.K1(hidden_states)))
            V1 = self.transpose_for_scores(self.act(self.V1(hidden_states)))
        elif self.attn_var == "KV2":
            K1 = V1 = self.transpose_for_scores(self.act(self.V1(hidden_states)))
        elif self.attn_var == "Ka2":
            K1 = self.transpose_for_scores(self.K1(self.act(hidden_states)))
            V1 = self.transpose_for_scores(self.V1(self.K1(hidden_states)))
        elif self.attn_var == "Ka3":
            K1 = self.transpose_for_scores(self.K1(hidden_states))
            V1 = self.transpose_for_scores(self.V1(self.K1(hidden_states)))
        elif self.attn_var == "Ka4":
            K1 = self.transpose_for_scores(self.K1(self.act(hidden_states)))
            V1 = self.transpose_for_scores(self.act(self.V1(hidden_states)))
        elif self.attn_var == "Ka5":
            K1 = self.transpose_for_scores(self.K1(hidden_states))
            V1 = self.transpose_for_scores(self.act(self.V1(hidden_states)))
        elif self.attn_var == "Ka6":
            K1 = self.transpose_for_scores(self.K1(self.act(hidden_states)))
            V1 = self.transpose_for_scores(self.V1(self.act(self.K1(hidden_states))))
        elif self.attn_var == "Ka7":
            K1 = self.transpose_for_scores(self.K1(hidden_states))
            V1 = self.transpose_for_scores(self.V1(self.act(self.K1(hidden_states))))
        elif self.attn_var == "Va2":
            K1 = self.transpose_for_scores(self.act(self.K1(hidden_states)))
            V1 = self.transpose_for_scores(self.V1(self.act(hidden_states)))
        elif self.attn_var == "Va3":
            K1 = self.transpose_for_scores(self.act(self.K1(hidden_states)))
            V1 = self.transpose_for_scores(self.V1(hidden_states))
        else:
            K1 = self.transpose_for_scores(self.act(self.K1(hidden_states)))
            V1 = self.transpose_for_scores(self.act(self.V1(hidden_states)))

        # V1 = self.transpose_for_scores(self.act(self.V1(hidden_states)))
        bs, length, n_head, hd = K1.shape
        dot_products = torch.einsum('blhn,hn->blh', K1, self.ReadingHead)

        if self.attn_var == "SM2": 
            valid_mask = self._softmax2(dot_products) > (2 / length)
        elif self.attn_var == "SM3": 
            valid_mask = self._softmax2(dot_products) > (1 / length)
        elif self.attn_var == "SM4": 
            valid_mask = self._softmax2(dot_products) > (.5 / length)
        elif self.attn_var == "SM5": 
            valid_mask = self._softmax2(dot_products) > (.1 / length)
        elif self.attn_var == "Sink1": 
            valid_mask = self._softmax2(dot_products) > (2 / length)
        elif self.attn_var == "Sink2": 
            valid_mask = self._softmax2(dot_products) > (1 / length)
        elif self.attn_var == "Sink3": 
            valid_mask = self._softmax2(dot_products) > (.5 / length)
        elif self.attn_var == "Sink4": 
            valid_mask = self._softmax2(dot_products) > (.2 / length)
        elif self.attn_var == "SV001": 
            valid_mask = dot_products > (0.01)
        elif self.attn_var == "SV01": 
            valid_mask = dot_products > (0.1)
        elif self.attn_var == "SV1": 
            valid_mask = dot_products > (1)
        else:
            valid_mask = self._softmax(dot_products) > (1.5 / length)
        new_states = torch.zeros_like(K1).to(DEVICE)

        forward_mmap = torch.full((bs, length, self.n_registers+1, n_head), 0, dtype=torch.long).to(DEVICE)
        backward_mmap = torch.full((bs, length, self.n_registers+1, n_head), 0, dtype=torch.long).to(DEVICE)

        forward_mmap[:, :, 0] = (torch.arange(1,1+length)*1.).view(1,length,1).expand(-1,-1,n_head)
        for len_index in range(1,length):
            forward_mmap[:, len_index, 1:] = torch.where(valid_mask[:, len_index].unsqueeze(1).expand(-1,self.n_registers,-1), forward_mmap[:, len_index-1, :-1], forward_mmap[:, len_index-1, 1:])
        forward_mmap = forward_mmap[:,:,1:].reshape(bs, length*self.n_registers, n_head, 1).expand(-1, -1, -1, hd)

        backward_mmap[:, :, 0] = (torch.arange(0,length)*1.).view(1,length,1).expand(-1,-1,n_head)
        backward_mmap[:, length - 1, 1] = torch.where(valid_mask[:, length - 1], length - 1, 0)
        for len_index in reversed(range(1,length-1)):
            backward_mmap[:, len_index, 1:] = torch.where(valid_mask[:, len_index].unsqueeze(1).expand(-1,self.n_registers,-1), backward_mmap[:, len_index+1, :-1], backward_mmap[:, len_index+1, 1:])
        backward_mmap = backward_mmap[:,:,1:].reshape(bs, length*self.n_registers, n_head, 1).expand(-1, -1, -1, hd)

        V_forward = torch.gather(V1, 1, forward_mmap).view(bs, length, self.n_registers, n_head, hd)
        V_backward = torch.gather(V1, 1, backward_mmap).view(bs, length, self.n_registers, n_head, hd)
        new_states = (torch.cat([V_forward, V_backward], dim=2).permute(0,1,3,4,2) * self.bidirection_weight).sum(dim=-1)
        new_states = new_states.reshape(bs, length, -1)

        if self.attn_var == "P1": new_states = self.act(self.P1(new_states))

        new_states = new_states.reshape(bs, length, -1)
        return (new_states, attention_probs) if output_attentions else (new_states,)

# Some good and bad designs

# Gates: Good
# G1 = self.transpose_for_scores(self.sigmoid(self.G1(hidden_states)))
# new_states = self.P1(hidden_states)
# context_layer = (G1 * new_states).reshape(bs, length, -1)

# nHeads: Bad
# Using more heads and P1 is not helpful.

# sigmoid(v1): Bad
# V1 = V1 * self.sigmoid(dot_products + self.bias_h).unsqueeze(3)

# update readhead: Bad
# 1. Not improving with subtraction: self.ReadingHead -= 0.001 * (K1 * (valid_mask==1).unsqueeze(3)).mean(dim=[0,1])
# 2. Worse with addition: self.ReadingHead += 0.001 * (K1 * (valid_mask==1).unsqueeze(3)).mean(dim=[0,1])
    
def replace_layer(model, args):
    for layer_index in range(1, args.n_blocks + 1):
        old_module = model.roberta.encoder.layer[layer_index-1].attention.self
        args.hidden_size = old_module.num_attention_heads * old_module.attention_head_size
        args.layer_index = layer_index
        model.roberta.encoder.layer[layer_index-1].attention.self = RobertaSelfAttention_matchKV(args)
        
    for layer_index in range(args.n_blocks + 1, 13):
        model.roberta.encoder.layer[layer_index-1].attention.self = RobertaSelfAttention_identity()

if 99 in args.layer_indices:
    print("Using Original Transformer Attention")
else:
    print("Using Modified Attention")
    replace_layer(model, args)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
    if training_args.do_eval and not is_torch_tpu_available()
    else None,
)

train_result = trainer.train()