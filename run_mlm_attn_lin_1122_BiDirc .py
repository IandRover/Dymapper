"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

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

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from utils_mlm.parser import get_HF_Args
parsera = get_HF_Args()
model_args, data_args, training_args = parsera.parse_json_file("./configs/mlm_baseline.json")

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--run_name', type=str, default="mlm_1122")
parser.add_argument('--from_scratch', type=int, default=0)
parser.add_argument('--attn_var', type=str, default="")
parser.add_argument('--n_blocks', type=int, default=12)
parser.add_argument('--n_registers', type=int, default=2)
parser.add_argument('--n_heads', type=int, default=12)
parser.add_argument('--head_dim', type=int, default=64)
parser.add_argument('--layer_indices', nargs='+', default=[100], type=int, help='an integer for the accumulator')
args, _ = parser.parse_known_args()

from utils_mlm.misc import misc
train_dataset, eval_dataset, model, data_collator, tokenizer, compute_metrics, preprocess_logits_for_metrics = misc(logging, training_args, data_args, model_args, logger, args)

if args.from_scratch == 0:
    args.run_name += f"_FT"
else:
    args.run_name += f"_RI"
    
if any(element in range(1,13) for element in args.layer_indices):
    args.run_name += f"_L{''.join(str(number) for number in args.layer_indices)}"
elif 99 in args.layer_indices:
    args.run_name += f"_bl"
    args.layer_indices = []
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

if args.n_registers in [1, 2,3,4]:
    args.run_name += f"_nR{args.n_registers}"
else:
    raise ValueError("Please specify valid number of registers")

if args.n_heads != 12:
    args.run_name += f"_nH{args.n_heads}"

if args.attn_var != "":
    args.run_name += f"_{args.attn_var}"

training_args.run_name = args.run_name
training_args.output_dir = os.path.join(training_args.output_dir, args.run_name)
if "mlm_1122" not in training_args.run_name: assert 0 == 1, f"Please make sure about the name of model"
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

        self.num_attention_heads = args.n_heads
        self.attention_head_size = args.head_dim
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.K1 = nn.Linear(args.hidden_size, self.all_head_size)
        self.V1 = nn.Linear(args.hidden_size, self.all_head_size)
        self.act = nn.ReLU(inplace=False)

        print(f"\t module type: {args.run_name}")
        self.run_name = args.run_name

        self.n_registers = args.n_registers
        self.bidirection_weight = nn.Parameter(torch.ones((1,1,self.num_attention_heads,1,self.n_registers*2))*(1/self.n_registers/2))
        self.ReadingHead = nn.Parameter(torch.zeros((self.num_attention_heads, self.attention_head_size)))
        
        self.attn_var = args.attn_var
        if self.attn_var == "softmax15" or self.attn_var == "softmax30":
            print("Using Softmax")
            self.softmax = torch.nn.Softmax(dim=1)
        if self.attn_var == "1softmax15" or self.attn_var == "1softmax30":
            print("Using One-Softmax")
        
        nn.init.xavier_normal_(self.ReadingHead)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size()[:-1] + (self.num_attention_heads, self.attention_head_size))

    def _softmax1(self, x):
        maxes = torch.max(x, 1, keepdim=True)[0]
        x_exp = torch.exp(x-maxes)
        return x_exp/torch.sum(x_exp, 1, keepdim=True) + 1 / torch.exp(maxes)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):

        DEVICE = hidden_states.device
        K1 = self.transpose_for_scores(self.act(self.K1(hidden_states)))
        V1 = self.transpose_for_scores(self.act(self.V1(hidden_states)))

        bs, length, n_head, hd = K1.shape
        dot_products = torch.einsum('blhn,hn->blh', K1, self.ReadingHead)
        if self.attn_var == "softmax15":
            valid_mask = self.softmax(dot_products) > (1.5 / length)
        elif self.attn_var == "softmax30":
            valid_mask = self.softmax(dot_products) > (3.0 / length)
        elif self.attn_var == "1softmax15":
            valid_mask = self._softmax1(dot_products) > (1.5 / length)
        elif self.attn_var == "1softmax30":
            valid_mask = self._softmax1(dot_products) > (3.0 / length)
        else:
            valid_mask = dot_products > 0.5
        new_states = torch.zeros_like(K1).to(DEVICE)
        
        forward_valids = torch.full((bs, self.n_registers+1, n_head), 0, dtype=torch.long).to(DEVICE)
        backward_valids = torch.full((bs, self.n_registers+1, n_head), 0, dtype=torch.long).to(DEVICE)
        forward_mmap = torch.full((bs, length, self.n_registers, n_head), 0, dtype=torch.long).to(DEVICE)
        backward_mmap = torch.full((bs, length, self.n_registers, n_head), 0, dtype=torch.long).to(DEVICE)

        for len_index in range(1,length):
            forward_valids[:, 0] = len_index 
            forward_valids[:, 1:] = torch.where(valid_mask[:, len_index].unsqueeze(1).expand(-1,self.n_registers,-1), forward_valids[:, :-1], forward_valids[:, 1:])
            forward_mmap[:, len_index] = forward_valids[:, 1:]
        forward_mmap = forward_mmap.view(bs, length*self.n_registers, n_head, 1).expand(-1, -1, -1, hd)

        for len_index in reversed(range(1,length)):
            backward_valids[:, 0] = len_index
            backward_valids[:, 1:] = torch.where(valid_mask[:, len_index].unsqueeze(1).expand(-1,self.n_registers,-1), backward_valids[:, :-1], backward_valids[:, 1:])
            backward_mmap[:, len_index] = backward_valids[:, 1:]
        backward_mmap = backward_mmap.view(bs, length*self.n_registers, n_head, 1).expand(-1, -1, -1, hd)

        V_forward = torch.gather(V1, 1, forward_mmap).view(bs, length, self.n_registers, n_head, hd)
        V_backward = torch.gather(V1, 1, backward_mmap).view(bs, length, self.n_registers, n_head, hd)
        new_states = torch.cat([V_forward, V_backward], dim=2).permute(0,1,3,4,2) * self.bidirection_weight    
        context_layer = new_states.sum(dim=-1).view(bs, length, -1)

        return (context_layer, attention_probs) if output_attentions else (context_layer,)

def replace_layer(model, args):
    for layer_index in args.layer_indices:
        from transformers import RobertaConfig
        old_module = model.roberta.encoder.layer[layer_index-1].attention.self
        args.hidden_size = old_module.num_attention_heads * old_module.attention_head_size
        print(f"Layer {layer_index}: ")
        model.roberta.encoder.layer[layer_index-1].attention.self = RobertaSelfAttention_matchKV(args)
        
def replace_layer_trimmed(model, args):
    for layer_index in range(1, args.n_blocks + 1):
        old_module = model.roberta.encoder.layer[layer_index-1].attention.self
        args.hidden_size = old_module.num_attention_heads * old_module.attention_head_size
        print(f"Layer {layer_index}: ")
        model.roberta.encoder.layer[layer_index-1].attention.self = RobertaSelfAttention_matchKV(args)
        
    for layer_index in range(args.n_blocks + 1, 13):
        print(f"Layer {layer_index}: set to identity")
        model.roberta.encoder.layer[layer_index-1].attention.self = RobertaSelfAttention_identity()
        
if args.n_blocks == 12: replace_layer(model, args)
else: replace_layer_trimmed(model, args)


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