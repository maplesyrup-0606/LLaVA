# Modifications

This README is to explain what modifications happened to the LLaVA model and how to utilize these modifications.

## `LLamaModel`

```python
def build_gaussian_connected_decay_mask(self, scanpoints, grid_size=24, sigma=1.5, margin=1, device='cuda'):

        mask = torch.zeros((1, 1, grid_size, grid_size), device=device)

        for x, y in scanpoints :
            x_start = max(0, x - margin)
            x_end = min(grid_size - 1, x + margin) 
            y_start = max(0, y - margin) 
            y_end = min(grid_size - 1, y + margin)
            mask[0, 0, y_start : y_end + 1, x_start : x_end + 1] = 1.0
    
        final_mask = torch.zeros_like(mask)
        visited = torch.zeros_like(mask, dtype=torch.bool) 

        max_steps = math.ceil(3 * sigma)

        kernel = torch.ones((1, 1, 3, 3), device=device)

        for d in range(max_steps + 1) :
            decay_weight = math.exp(- (d ** 2) / (2 * sigma ** 2))
            new_layer = (mask > 0) & (~visited)
            final_mask[new_layer] = decay_weight
            visited = visited | new_layer

            mask = (F.conv2d(mask, kernel, padding = 1) > 0).float()
        
        return final_mask.view(grid_size * grid_size).squeeze(0).squeeze(0)
```

This function creates a gaussian mask for each scanpath within the batch of images.

It first creates a connected binary mask of 1's and applies a gaussian decay up-to `max_steps`.

A sample image would look like this,
<p float="left">
  <img src="image.png" alt="alt text" width="45%" />
  <img src="image-1.png" alt="alt text" width="45%" />
</p>

These Gaussian Masks are passed in to all Decoder Layers.

## `LLamaAttention`
First, before we call `generate` we can create a `mask_type` which indicates what type of mask to apply to the attention weights.

It is consisted of the parameters :
- `type`
    - [`gaussian`](#gaussian-method) : Plain Gaussian Method
    - [`non-gaussian`](#non-gaussian-method) : Binary Masking Method
    - [`salient-head`](#salient-head-method) : Salient Head Method
- `margin` : How much of a margin should each gaze point have. E.g. `margin=1` would create a 3 x 3 window around the gaze point.
- `target-layer` : Which layer to target. Can be `None` if applied on all layers.

#### Gaussian Method
---
```python
def apply_gaussian_to_weights(self, attn_weights, image_infos, scanpaths, sigma=4.0, grid_size=24) :
        
    x_coords, y_coords = torch.arange(grid_size, device=attn_weights.device), torch.arange(grid_size, device=attn_weights.device)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
    X, Y = X.unsqueeze(0), Y.unsqueeze(0)
    
    for b, image_info in enumerate(image_infos) :
        start_idx = image_info['start_index']
        end_idx = start_idx + image_info['num_patches']

        if len(scanpaths) == 0 : continue

        gaze = torch.tensor(scanpaths[b], device=attn_weights.device)
        gx, gy = gaze[:, 0].view(-1, 1, 1), gaze[:, 1].view(-1, 1, 1)
        g = torch.exp(-((X - gx) ** 2 + (Y - gy) ** 2) / (2 * sigma ** 2))
        gaussian = g.sum(dim=0)

        gaussian = gaussian / (gaussian.max() + 1e-6)
        gaussian = gaussian.reshape(1, 1, -1)

        attn_weights[b, :, :, start_idx:end_idx] *= gaussian

    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
    return attn_weights
```

We basically build a Gaussian Map where the peaks locate at the gaze points and multiply the guassian map directly to the attention weights (for all query tokens).
#### Non-Gaussian Method
---
```python
def apply_gaze_mask(self, attn_weights, image_infos, scanpaths, grid_size=24, margin=0) :
        
    if not any(len(sp) > 0 for sp in scanpaths) :
        return attn_weights
    
    custom_mask = attn_weights.clone()
    B, H, Q, K = custom_mask.shape
    eps = 1e-6

    for b, image_info in enumerate(image_infos) :
        if len(scanpaths[b]) == 0 : 
            continue

        start_idx = image_info['start_index']
        L = image_info['num_patches']
        end_idx = start_idx + L
        
        
        offsets = torch.tensor([
            (dx, dy)
            for dx in [-margin, 0, margin]
            for dy in [-margin, 0, margin]
        ], device=attn_weights.device, dtype=torch.long)

        gaze = torch.tensor(scanpaths[b], device=attn_weights.device, dtype=torch.long)
        if gaze.numel() == 0:
            continue

        neighbors = (gaze[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        neighbors[:, 0].clamp_(0, grid_size - 1)
        neighbors[:, 1].clamp_(0, grid_size - 1)
        unique_neighbors = torch.unique(neighbors, dim=0)

        target_idx = start_idx + unique_neighbors[:, 0] + unique_neighbors[:, 1] * grid_size
        target_idx = target_idx.clamp(start_idx, end_idx - 1).unique()

        keep_k = torch.ones(K, dtype=torch.bool, device=attn_weights.device)
        keep_k[start_idx:end_idx] = False
        keep_k[target_idx] = True
        keep_k = keep_k.to(custom_mask.dtype)

        custom_mask[b] = custom_mask[b] * keep_k[None, None, :]

        row_sum = custom_mask[b].sum(dim=-1, keepdim=True).clamp_min(eps)
        custom_mask[b] = custom_mask[b] / row_sum

    return custom_mask
```
This method zeros-out the region that the scanpath doesn't lie and keeps only the gaze points + margin for the attention weights.
#### Salient-Head Method
---
```python
TARGET_LAYERS = range(19, 27)
        
tau = 0.5232
k = 8
sel = None
# attn_weights = bsz x heads x q_len x head_dim
if (self.layer_idx in TARGET_LAYERS) and (gaussian_masks is not None) and (mask_type["type"] == "salient-head") :
    img_start, L = 35, 576
    img_slice = slice(img_start, img_start + L)

    mask_stack = torch.stack([
        m if m is not None else torch.zeros(L, device=attn_weights.device, dtype=attn_weights.dtype)
        for m in gaussian_masks
    ])

    vis = attn_weights[:, :, -1, img_slice] # bsz x num_heads x L

    mask_exp = mask_stack.unsqueeze(1).expand(-1, self.num_heads, -1) # bsz x num_heads x L

    num = (vis * mask_exp).sum(dim=2) # bsz x num_heads
    den = vis.sum(dim=2).clamp_min(1e-9) # bsz x num_heads
    sim = num / den # bsz x num_heads

    # relative scaling per layer
    layer_max = sim.max(dim=1, keepdim=True).values # bsz x 1 
    sim = sim / (layer_max + 1e-9) # bsz x num_heads

    above = sim > tau # bsz x num_heads

    if not above.any() :
        pass
    
    scores = sim.masked_fill(~above, float("-inf")) 
    topk_idx = scores.topk(k, dim=1).indices # bsz x k
    sel = torch.zeros_like(above, dtype=torch.bool) # bsz x num_heads
    sel.scatter_(1, topk_idx, True)
    sel &= above

    sel_mask = sel.unsqueeze(-1) # bsz x num_heads x 1 
    vis = torch.where(sel_mask, vis * mask_exp, vis)
    attn_weights[:, :, -1, img_slice] = vis

    eps = 1e-6
    last = attn_weights[:, :, -1, :].float()
    if sel.any():
        rows = last[sel]
        rows_sum = rows.sum(dim=-1, keepdim=True).clamp_min(eps)
        last[sel] = rows / rows_sum

    attn_weights[:, :, -1, :] = last.to(attn_weights.dtype)
```

For this method, based on the gaussian masks ($G$) we first compute the similarity score for each head,
$$\text{Sim}({h_{\ell}}) = \frac{\sum A_\text{vis} \otimes G}{\sum A_\text{vis}}$$
and based on the $\text{Sim}$ score, we choose the **salient-heads** and multiply the gaussian mask directly to the Attention Weights (following a normalization).
