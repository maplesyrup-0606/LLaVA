import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from io import BytesIO
from transformers import TextStreamer
import numpy as np
import matplotlib.pyplot as plt

def process_scanpaths(scanpaths) :
    """
    scanpaths: scanpaths per image rescaled to 336 x 336 for each image.

    We convert each point in the scanpath to match the 24 x 24 grid 
    of patches (in indices). 
    """

    for scanpath in scanpaths :
        scanpath['X'] =  (scanpath['X'] // 24).astype(int)
        scanpath['Y'] = (scanpath['Y'] // 24).astype(int)
    
    return scanpaths

def plot_tensor_with_scanpath(image_tensor, scanpath, save_path, unnormalize=True):
    """
    image_tensor: torch.Tensor of shape [3, H, W]
    scanpath: dict with 'X' and 'Y' (in pixel coordinates)
    save_path: output file path (e.g. "out/scanpath_0.png")
    unnormalize: whether to unnormalize the image for plotting
    """
    img = image_tensor.clone()

    if unnormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean

    img_np = img.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    img_np = np.clip(img_np, 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(img_np)
    
    xs, ys = scanpath['X'], scanpath['Y']
    ax.plot(xs, ys, marker='o', color='red', linewidth=1.5, alpha=0.6)

    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, str(i+1), color='yellow', fontsize=8, ha='center', va='center')

    ax.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    if args.scanpath is not None :
        scanpaths = [np.load(os.path.expanduser(args.scanpath), allow_pickle=True).item()]

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    image_size = image.size


    # Similar operation in model_worker.py
    image_tensor, new_scanpaths = process_images([image], image_processor, model.config, scanpaths)

    # # NOTE:Sanity check to see if the resized scanpaths match
    # for i in range(len(image_tensor)): 
    #     save_path = f"scanpath_plots/scanpath_{i}.png"
    #     plot_tensor_with_scanpath(image_tensor[i], new_scanpaths[i], save_path)
    # print("done saving!")

    processed_scanpaths = process_scanpaths(new_scanpaths)

    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                scan_paths=processed_scanpaths,
                output_attentions=True, # NOTE: Added to visualize attention
                return_dict=True,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)


        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--scanpath", type=str, default="~/NSERC/scanpath.npy")
    args = parser.parse_args()
    main(args)
