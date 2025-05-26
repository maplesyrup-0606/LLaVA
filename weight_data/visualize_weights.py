import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
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
    # image_path = os.path.expanduser('~/NSERC/room.jpg')
    image_path = os.path.expanduser(image_path)
    data_path = os.path.expanduser(data_path)
    
    name = os.path.splitext(os.path.basename(data_path))[0] + ".png"
    image = Image.open(image_path)
    data = torch.load(data_path)

    image_info = data['image_infos']

    start_idx, end_idx = image_info[0]['start_index'], image_info[0]['start_index'] + image_info[0]['num_patches']
    side_len = 24
    mean_rgb = tuple(map(int, np.array(image).mean(axis=(0, 1))))
    resized_image = expand2square(image, mean_rgb)

    attn_weights = data['attentions'] # this is huge : num_out_tokens x layers x (bsz x heads x q_len x num_tokens)

    d1 = len(attn_weights)
    d2 = len(attn_weights[0])

    processed_groups = []
    for outer_tuple in attn_weights :
        group_tensors = []
        for tensor in outer_tuple :
            if tensor.shape[2] != 1 :
                tensor = tensor[..., -1:, :]
            tensor = tensor[..., start_idx : end_idx]
            group_tensors.append(tensor)
        
        stacked_group = torch.stack(group_tensors, dim=0)
        processed_groups.append(stacked_group)
        
    attn_weights = torch.stack(processed_groups, dim=0) # 10 x 32 x 1 x 32 x 1 x 576, only image_tokens 
    attn_weights = attn_weights[:, layer, ...] # 10 x 1 x 1 x 32 x 1 x 576
    attn_weights = attn_weights.squeeze(1).squeeze(1).squeeze(2) # 10 x 32 x 576
    attn_weights_avg = attn_weights.mean(dim=(0,1))

    attn_grid = attn_weights_avg.view((side_len, side_len))
    attn_grid = attn_grid / attn_grid.sum()
    attn_grid = attn_grid.detach().cpu().numpy()

    attn_tensor = np.expand_dims(np.expand_dims(attn_grid, 0), 0)  # shape: (1, 1, H, W)
    attn_tensor = torch.tensor(attn_tensor, dtype=torch.float32)
    attn_unsampled = F.interpolate(attn_tensor, size=resized_image.size, mode='bilinear', align_corners=False)
    attn_map = attn_unsampled.squeeze().numpy()

    image_np = np.array(resized_image)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(attn_map, cmap='jet', alpha=0.5)  # alpha for transparency
    plt.axis('off')
    plt.title(f"Layer {layer}")
    plt.savefig(f'{layer}_{name}', dpi=300)
    plt.show()

if __name__ == '__main__' :
    # visualize_attention("~/NSERC/images_captions_for_test/COCO_train2014_000000513461.jpg", "~/NSERC/LLaVA/weight_data/image_2_gaussian.pt", 14)
    visualize_attention("~/NSERC/images_captions_for_test/COCO_train2014_000000513461.jpg", "~/NSERC/LLaVA/weight_data/image_2_without_gaussian.pt", 14)
    # image_weight_pair = [
    #     ("~/NSERC/images_captions_for_test/COCO_train2014_000000318556.jpg", "~/NSERC/LLaVA/weight_data/COCO/COCO_Image_1.pt", 14),
    #     ("~/NSERC/images_captions_for_test/COCO_train2014_000000318556.jpg", "~/NSERC/LLaVA/weight_data/COCO/COCO_Image_2.pt", 14),
    #     ("~/NSERC/images_captions_for_test/COCO_train2014_000000318556.jpg", "~/NSERC/LLaVA/weight_data/COCO/COCO_Image_3.pt", 14),

    #     ("~/NSERC/images_captions_for_test/COCO_train2014_000000513461.jpg", "~/NSERC/LLaVA/weight_data/COCO/COCO_Image_1_image2.pt", 14),
    #     ("~/NSERC/images_captions_for_test/COCO_train2014_000000513461.jpg", "~/NSERC/LLaVA/weight_data/COCO/COCO_Image_2_image2.pt", 14),

    #     ("~/NSERC/images_captions_for_test/COCO_train2014_000000539984.jpg", "~/NSERC/LLaVA/weight_data/COCO/COCO_Image_1_image3.pt", 14),
    #     ("~/NSERC/images_captions_for_test/COCO_train2014_000000539984.jpg", "~/NSERC/LLaVA/weight_data/COCO/COCO_Image_2_image3.pt", 14),
    # ]

    # for image_path, data_path, layer in image_weight_pair :
    #     visualize_attention(image_path=image_path, data_path=data_path, layer=layer)