"""
Implementation of a GPT-2 model taken from https://github.com/karpathy/nanoGPT/blob/master/model.py

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from src.pipelines.shared.base_models import BaseNightingaleModel
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, model_config: dict):
        super().__init__()
        assert model_config["model_dim"] % model_config["n_heads"] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(model_config["model_dim"], 3 * model_config["model_dim"], bias=model_config["bias"])
        # output projection
        self.c_proj = nn.Linear(model_config["model_dim"], model_config["model_dim"], bias=model_config["bias"])
        # regularization
        self.attn_dropout = nn.Dropout(model_config["dropout"])
        self.resid_dropout = nn.Dropout(model_config["dropout"])
        self.n_head = model_config["n_heads"]
        self.n_embd = model_config["model_dim"]
        self.dropout = model_config["dropout"]
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(model_config["context_length"], model_config["context_length"]))
                                        .view(1, 1, model_config.context_length, model_config.context_length))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, model_config: dict):
        super().__init__()
        self.c_fc    = nn.Linear(model_config["model_dim"], 4 * model_config["model_dim"], bias=model_config["bias"])
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * model_config["model_dim"], model_config["model_dim"], bias=model_config["bias"])
        self.dropout = nn.Dropout(model_config["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, model_config: dict):
        super().__init__()
        self.ln_1 = LayerNorm(model_config["model_dim"], bias=model_config["bias"])
        self.attn = CausalSelfAttention(model_config)
        self.ln_2 = LayerNorm(model_config["model_dim"], bias=model_config["bias"])
        self.mlp = MLP(model_config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(BaseNightingaleModel):
    """
    Implementation of a GPT-2 model (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
    adapted from https://github.com/karpathy/nanoGPT

    Args:
        vocab_size (int): The size of the vocabulary.
        model_dim (int): The dimension of the model.
        n_layers (int): The number of layers in the model.
        dropout (float): The dropout rate.
        n_heads (int): The number of attention heads.
        context_length (int): The length of the context.
        bias (bool): Whether to use bias in the model.
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.model_dim = model_config["model_dim"]
        self.n_layers = model_config["n_layers"]
        self.context_length = model_config["context_length"]

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(model_config["vocab_size"], model_config["model_dim"]),
            wpe = nn.Embedding(model_config["context_length"], model_config["model_dim"]),
            drop = nn.Dropout(model_config["dropout"]),
            h = nn.ModuleList([Block(model_config) for _ in range(model_config["n_layers"])]),
            ln_f = LayerNorm(model_config["model_dim"], bias=model_config["bias"]),
        ))
        self.lm_head = nn.Linear(model_config["model_dim"], model_config["vocab_size"], bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * model_config["n_layers"]))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def required_config_keys(self) -> set[str]:
        return {"vocab_size", "model_dim", "n_layers", "dropout", "n_heads", "context_length"}

    def required_input_keys(self) -> set[str]:
        return {"ehr.input_token_ids"}

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: dict, targets=None):

        self.validate_input(x)

        input_token_ids = x["ehr"]["input_token_ids"]
        device = input_token_ids.device
        b, t = input_token_ids.size()
        assert t <= self.context_length, f"Cannot forward sequence of length {t}, block size is only {self.context_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_token_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

if __name__ == "__main__":
    # define params
    batch_size = 1
    sequence_length = 10
    vocab_size = 100
    num_heads = 1
    model_dim = num_heads * 64
    n_layers = 2
    dropout = 0.5
    bias = True

    # random input
    rand = torch.randint(0, vocab_size, (batch_size, sequence_length + 1))
    x = {
        "ehr": {
            "input_token_ids": rand[:, :-1],
            "target_token_ids": rand[:, 1:]
        }
    }

    model_config = {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "n_heads": num_heads,
        "context_length": sequence_length,
        "bias": bias,
    }
    model = GPT2(model_config)
    print(model)
    print(x)
    output = model(x)
    print(output.shape)