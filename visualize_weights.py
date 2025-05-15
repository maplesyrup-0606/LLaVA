import torch
import matplotlib.pyplot as plt
import numpy as np
attn_weights = torch.load('attention_weights.pt')
# Convert 32-tuple → tensor of shape (32, 1, 32, 1, 748)
attn_stack = torch.stack(attn_weights)  # (layers, batch, heads, query_len, key_len)
attn_stack = attn_stack.squeeze(1).squeeze(2).squeeze(2)  # (32, 32, 748)

layer = 0
head = 0

weights = attn_stack[layer, head].detach().cpu().numpy()

plt.figure(figsize=(12, 3))
plt.plot(weights)
plt.title(f"Attention Weights - Layer {layer}, Head {head}")
plt.xlabel("Key Token Index")
plt.ylabel("Attention Weight")
plt.tight_layout()
plt.savefig(f"attn_plot_layer{layer}_head{head}.png", dpi=300)
plt.close()