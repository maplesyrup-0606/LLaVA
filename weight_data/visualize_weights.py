import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image

def expand2square(pil_img, background_color) :
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def visualize_attention(image_path, data_path, layer) :
    # image = Image.open('../room.jpg')
    # data = torch.load('weights_and_image.pt')
    name = data_path[:-3]
    image = Image.open(image_path)
    data = torch.load(data_path)


    mean_rgb = tuple(map(int, np.array(image).mean(axis=(0, 1))))
    resized_image = expand2square(image, mean_rgb)

    attn_weights = data['attn_weights']
    image_info = data['image_infos'][0]

    start_idx, end_idx = image_info['start_index'], image_info['start_index'] + image_info['num_patches']
    side_len = 24
    # Convert 32-tuple → tensor of shape (32, 1, 32, 1, 748)
    attn_stack = torch.stack(attn_weights)  # (layers, batch, heads, query_len, key_len)
    attn_stack = attn_stack.squeeze(1).squeeze(2).squeeze(2)  # (32, 32, 748)

    head = 0

    weights = attn_stack[layer].mean(dim=0).detach().cpu().numpy()
    image_patch_weights = weights[start_idx: end_idx]
    attn_grid = image_patch_weights.reshape((side_len, side_len))
    attn_grid = attn_grid / attn_grid.max()

    attn_tensor = np.expand_dims(np.expand_dims(attn_grid, 0), 0)  # shape: (1, 1, H, W)
    attn_tensor = torch.tensor(attn_tensor, dtype=torch.float32)
    attn_unsampled = F.interpolate(attn_tensor, size=resized_image.size, mode='bilinear', align_corners=False)
    attn_map = attn_unsampled.squeeze().numpy()

    image_np = np.array(resized_image)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(attn_map, cmap='jet', alpha=0.5)  # alpha for transparency
    plt.axis('off')
    plt.title(f"Attention Overlay - Layer {layer}, Head {head}")
    plt.savefig(f'{layer}_{name}', dpi=300)
    plt.show()

if __name__ == '__main__' :
    # visualize_attention('../room.jpg', 'weights_and_image.pt', 7)
    visualize_attention('../room.jpg', 'weights_and_image_specific_question.pt', 14)
    visualize_attention('../room.jpg', 'weights_and_image_specific_question_without_custom.pt', 14)