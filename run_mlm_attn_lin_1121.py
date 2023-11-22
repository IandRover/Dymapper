"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import os
import logging
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    Trainer,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.36.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.

# parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))


# model_args, data_args, training_args = parser.parse_json_file("./configs/mlm.json")
from utils_mlm.parser import get_HF_Args
parsera = get_HF_Args()
model_args, data_args, training_args = parsera.parse_json_file("./configs/mlm_baseline.json")

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--run_name', type=str)
parser.add_argument('--from_scratch', type=int, default=0)
parser.add_argument('--attn_var', type=str, default="")
parser.add_argument('--n_heads', type=int, default=12)
parser.add_argument('--head_dim', type=int, default=64)
parser.add_argument('--layer_indices', nargs='+', default=[99], type=int, 
                    help="""1-13 indicates the layer indices;
                            99 means baseline: no attention replacement; 
                            100 means all layers""")
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
    # 100 means baseline: no attention replacement
    args.run_name += f"_bl"
    args.layer_indices = []
elif 100 in args.layer_indices:
    # 100 means all layers
    args.run_name += f"_Lall"
    args.layer_indices = range(1,13)
else:
    raise ValueError("Please specify the layer indices")

if args.n_heads != 12:
    args.run_name += f"_H{args.n_heads}"


training_args.run_name = args.run_name
training_args.output_dir = os.path.join(training_args.output_dir, args.run_name)
print(f"Session Name: {training_args.run_name}")
print(f"Output Dir: {training_args.output_dir}")
# ========== NEW FEATURE: attn_conv_val ==========
import torch
import torch.nn as nn

class RobertaSelfAttention_matchKV(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_attention_heads = args.n_heads
        self.attention_head_size = args.head_dim
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)
        self.act = nn.ReLU(inplace=False)

        print(f"\t module type: {args.run_name}")
        self.run_name = args.run_name

        self.w1 = nn.Parameter(torch.ones((1))*0.7)
        self.w2 = nn.Parameter(torch.ones((1))*0.3)
        self.ReadingHead = nn.Parameter(torch.zeros((self.num_attention_heads, self.attention_head_size)))
        
        nn.init.xavier_normal_(self.ReadingHead)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size()[:-1] + (self.num_attention_heads, self.attention_head_size))
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, 
                encoder_attention_mask=None, past_key_value=None, output_attentions=False):

        # Hidden States: Batch, seq, hidden_dim (512)
        # Key Layer: Batch, seq, n_head (12), hidden_dim (64)
        K = self.transpose_for_scores(self.act(self.key(hidden_states)))
        K = K.permute(0,2,1,3)
        V1 = self.transpose_for_scores(self.act(self.value(hidden_states)))
        V1 = V1.permute(0,2,1,3)

        bs, length, n_head, hd = K.shape
        dot_products = torch.einsum('blnh,nh->bln', K, self.ReadingHead)
        valid_mask = dot_products > 0.5
        new_states = torch.zeros_like(K).to(K.device)

        last_two_valid = torch.full((bs, n_head, 2), 0, dtype=torch.long).to(K.device)
        valid_indices = torch.full((bs, n_head, length, 2), 0, dtype=torch.long).to(K.device)

        # Update last_two_valid and valid_indices without using the inefficient inner loop
        for len_index in range(length):
            current_valid_mask = valid_mask[:, len_index, :]
            last_two_valid[:, :, 1] = torch.where(current_valid_mask, last_two_valid[:, :, 0], last_two_valid[:, :, 1])
            last_two_valid[:, :, 0] = torch.where(current_valid_mask, len_index, last_two_valid[:, :, 0])
            valid_indices[:, :, len_index, 0] = last_two_valid[:, :, 0]
            valid_indices[:, :, len_index, 1] = last_two_valid[:, :, 1]

        assert torch.max(valid_indices) < V1.shape[1], "Index out of bounds: index is too large"
        assert torch.min(valid_indices) >= 0, "Index out of bounds: index is negative"
        
        valid_indices = valid_indices.view(bs, n_head, length*2).permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, hd) # > (bs, length*2, n_head, hd)

        new_states = torch.gather(V1, 1, valid_indices).reshape(bs, length, 2, n_head, hd)
        new_states = new_states[:,:,0] * self.w1 + new_states[:,:,1] * self.w2
        context_layer = new_states.view(bs, length, -1).contiguous()

        # Assuming attention_probs are calculated elsewhere and available
        
        return (context_layer, attention_probs) if output_attentions else (context_layer,)

def replace_layer(model, args):
    for layer_index in args.layer_indices:
        from transformers import RobertaConfig
        old_module = model.roberta.encoder.layer[layer_index-1].attention.self
        args.hidden_size = old_module.num_attention_heads * old_module.attention_head_size
        print(f"Layer {layer_index}: ")
        model.roberta.encoder.layer[layer_index-1].attention.self = RobertaSelfAttention_matchKV(args)
        
replace_layer(model, args)

if "mlm_1121" in training_args.run_name:
    pass
else:
    assert 0 == 1, f"Please make sure about the name of model"

# ========== NEW FEATURE: attn_conv_val ==========


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