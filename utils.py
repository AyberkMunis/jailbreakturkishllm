import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List
from captum.attr import LayerConductance
def compute_Ia_all_layers(
    model,
    tokenizer,
    prompt: str,
    object_token_str: str,
    token_position: int = -1,
    top_k: int = 100,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):

    model.eval()
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    object_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(object_token_str)[0])
    e_o = model.lm_head.weight[object_token_id].to(device).to(torch.float32)

    all_attn_outputs = []

    hooks = []
    for l in range(len(model.model.layers)):
        def get_hook(layer_idx):
            def hook_fn(module, input, output):
                all_attn_outputs.append(output.detach())
            return hook_fn
        hook = model.model.layers[l].self_attn.o_proj.register_forward_hook(get_hook(l))
        hooks.append(hook)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for hook in hooks:
        hook.remove()

    E = model.lm_head.weight.to(device).to(torch.float32)

    I_a_scores = []

    for l in tqdm(range(len(model.model.layers)), desc="Computing I_a^{(l)}(o)"):
        a_l = all_attn_outputs[l][0, token_position, :].to(torch.float32)

        z_l = outputs.hidden_states[l + 1][0, token_position, :].to(torch.float32)

        logits = torch.matmul(E, z_l) 
        top_indices = torch.topk(logits, top_k + 1).indices.tolist()

        top_wrong_ids = [i for i in top_indices if i != object_token_id][:top_k]
        e_o_primes = E[top_wrong_ids]

        e_o_bar = torch.mean(e_o_primes, dim=0)

        # I_a^{(l)}(o) = a_l^T (e_o - mean(e_o'))
        score = torch.dot(a_l, e_o - e_o_bar).item()
        I_a_scores.append(score)

    return I_a_scores


def compute_logit_lens_across_layers(
    model,
    tokenizer,
    prompt: str,
    subject_token: str,
    object_token_str: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> List[float]:

    model.to(device)
    model.eval()

    # Tokenize giriş
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    # Konu tokeninin pozisyonunu bul
    subject_token_position=torch.where(input_ids==tokenizer.encode(subject_token)[-1])[-1].item()

    # Unembedding (lm_head weight) üzerinden object vektörü
    object_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(object_token_str)[0])
    e_o = model.lm_head.weight[object_token_id].to(device).to(torch.float32)

    # Hook'tan aktivasyonları toplayacak dict
    mlp_outputs = {}

    def get_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            mlp_outputs[layer_idx] = output.detach()
        return hook_fn

    # Bütün katmanlara hook bağla
    hooks = []
    num_layers = len(model.model.layers)
    for layer_idx in range(num_layers):
        hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(get_hook_fn(layer_idx))
        hooks.append(hook)

    # Modeli çalıştır
    with torch.no_grad():
        _ = model(**inputs)

    # Hookları kaldır
    for hook in hooks:
        hook.remove()

    # Logit lens skorlarını hesapla
    scores = []
    for layer_idx in range(num_layers):
        if layer_idx not in mlp_outputs:
            scores.append(float('nan'))
            continue

        m_s_l = mlp_outputs[layer_idx][0, subject_token_position, :].to(torch.float32)
        hidden_dim = m_s_l.shape[-1]
        layernorm = nn.LayerNorm(hidden_dim).to(device)
        m_s_l_norm = layernorm(m_s_l)

        I_m_l_o = torch.dot(e_o, m_s_l_norm).item()
        scores.append(I_m_l_o)

    return scores
def FullLayerConductance(model,tokenizer,prompt,target_token):
  inputs = tokenizer(prompt, return_tensors="pt")
  input_ids = inputs["input_ids"]
  with torch.no_grad():
    input_embeds = model.model.embed_tokens(input_ids)

  input_embeds.requires_grad_(True)

  target_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_token))
  attributions_per_layer = []
  layer_names = []
  def forward_func(input_embeds,target_id):
      outputs = model(inputs_embeds=input_embeds)
      logits = outputs.logits
      return logits[:, -1, target_id]
  for x,layer in enumerate(model.model.layers):
    mlp=layer.mlp
    layer_names.append(f"Layer {x} MLP")
    print(f"Layer {x} MLP")
    lc = LayerConductance(forward_func, mlp)
    token_id_score=[]
    for i in target_token_id:
      attributions = lc.attribute(inputs=input_embeds,additional_forward_args=(i,))
      token_id_score.append(attributions.sum().item())
    attributions_per_layer.append(sum(token_id_score)/len(token_id_score))
    print(f"Score:{sum(token_id_score)/len(token_id_score)}")
  return attributions_per_layer,layer_names


