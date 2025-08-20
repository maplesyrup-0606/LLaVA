import argparse
import torch
import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys


from pathlib import Path
from tqdm import tqdm
from PIL import Image
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, process_scanpaths, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from transformers import TextStreamer

BATCH_SIZE = 8

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

def load_image(image_file) :
    image = Image.open(image_file).convert('RGB')
    return image

def prepare_mask_type_dict(type_=None, margin=0, target_layer=None) :
    mask_type = {}

    if type_ is None :
        mask_type["type"] = "None"
    else :
        mask_type["type"] = type_
    
    mask_type["margin"] = margin

    if target_layer is not None :
        mask_type["target-layer"] = target_layer

    return mask_type

def main(args) :
    disable_torch_init()

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("CUDA device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "None", flush=True)
    # model preparation
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, 
                                                              args.model_base, 
                                                              model_name, 
                                                              args.load_8bit, 
                                                              args.load_4bit, 
                                                              device=args.device)
    
    print("Model device:", model.device, flush=True)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower(): # NOTE: This is the model we use
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    
    # if "mpt" in model_name.lower():
    #     roles = ('user', 'assistant')
    # else:
    #     roles = conv.roles

    # load scanpaths from coco-captions
    if args.scanpath is not None :
        scanpaths_dir = args.scanpath
    
    if args.images_dir is not None :
        images_dir = args.images_dir
    
    if args.weights_dir is not None :
        weights_dir = args.weights_dir

    if args.trajectory is not None :
        mode = args.trajectory

    mask_type = prepare_mask_type_dict(type_=args.type, margin=args.margin, target_layer=args.target_layer)
    print(mask_type)
    print(f"✅ Running Job, with parameters \n scanpaths : {scanpaths_dir} \n images : {images_dir} \n weights : {weights_dir}")


    image_root = Path(images_dir).expanduser()
    
    scan_root = Path(scanpaths_dir).expanduser()
    
    weights_dir = Path(weights_dir).expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    answers_path = Path(args.answers_file).expanduser()
    answers_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_root.glob("*/*.jpg"))

    if args.num_samples != -1 and args.num_samples < len(image_paths) :
        rng = np.random.default_rng(args.seed)
        image_paths = rng.choice(image_paths, size=args.num_samples, replace=False).tolist()

    image_ids, id_to_path = [], {}
    for p in image_paths:
        classdir = p.relative_to(image_root).parent.name
        long_id = f"{classdir}/{p.stem}"
        image_ids.append(long_id)
        id_to_path[long_id] = p

    if len(image_ids) == 0 :
        print("No images")
        return

    # Original Prompts
    # PROMPTS = [
    #     "Describe the image concisely in one sentence, up to 20 words only.",
    #     "Render a clear and concise summary of the photo in one sentence, up to 20 words only.",
    #     "Share a concise interpretation of the image provided in one sentence, up to 20 words only.",
    #     "Give a brief description of the image in one sentence, up to 20 words only.",
    # ]

    # New Prompt to avoid collapsing for old methods
    # PROMPTS = [
    #     "Please describe this image in detail, up to 65 words only."
    # ]

    # New Prompt
    PROMPTS = [
        "Please describe this image in detail."
    ]

    # Prompt forced for CUB_200_2011
    # PROMPTS = [
    #     "Describe the bird concisely in one sentence, up to 20 words only.",
    #     "Render a clear and concise summary of the bird in the photo in one sentence, up to 20 words only.",
    #     "Share a concise interpretation of the bird in the provided image in one sentence, up to 20 words only.",
    #     "Give a brief description of bird in the image in one sentence, up to 20 words only.",
    # ]

    output_captions: dict[str , list[str]] = defaultdict(list)

    for cap_k, prompt_tpl in enumerate(PROMPTS):
        total_batches = (len(image_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_start in tqdm(
            range(0, len(image_ids), BATCH_SIZE),
            total=total_batches,
            ascii=True,
            file=sys.stdout
        ):

            batch_img_ids = image_ids[batch_start : batch_start + BATCH_SIZE]
            
            prompts, all_images, all_scanpaths, image_sizes = (
                [],
                [],
                [],
                [],
            )
            for image_id in batch_img_ids :
                img_path = id_to_path[image_id]
                scan_path = scan_root / f"{image_id}_scanpath.npy"
                conv = conv_templates[conv_mode].copy()
                prompt_body = DEFAULT_IMAGE_TOKEN + "\n" + prompt_tpl
                conv.append_message(conv.roles[0], prompt_body)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())

                pil_img = load_image(img_path)
                image_sizes.append(pil_img.size)
                all_images.append(pil_img)

                if scan_path.exists() :
                    scanpaths = [np.load(scan_path, allow_pickle=True).item()]
                else :
                    scanpaths = [[]]
                
                all_scanpaths.append(scanpaths)
            
            
            image_tensor, new_scanpaths = process_images(all_images, image_processor, model.config, all_scanpaths)
            processed_scanpaths = process_scanpaths(new_scanpaths, mode=mode)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in prompts], dim=0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        scanpaths=processed_scanpaths,
                        mask_type=mask_type,
                        output_attentions=False,
                        return_dict_in_generate=True,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True)
                
                outputs = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)
        
            for j, image_id in enumerate(batch_img_ids):
                output_captions[image_id].append(outputs[j])

            save_helper(answers_path=answers_path, output_captions=output_captions)
            # if cap_k == 0 and batch_start < 8 :
            #     weights = output_ids['attentions']
            #     image_infos = output_ids['image_infos']
            #     data = {
            #         'attn_weights' : weights,
            #         'image_infos' : image_infos,
            #         'all_image_ids' : batch_img_ids,
            #         'scanpaths' : processed_scanpaths
            #     }
            #     torch.save(data, os.path.join(weights_dir, f"{batch_start // BATCH_SIZE}_weights.pt"))
            #     print(f"Saved weights to {weights_dir} ✅", flush=True)
            #     print(f"Sample outputs : {outputs}")
                

    save_helper(answers_path=answers_path, output_captions=output_captions)

    return

def save_helper(answers_path, output_captions):
    with answers_path.open("w", encoding="utf-8") as f:
        json.dump(output_captions, f, indent=2, ensure_ascii=False)
    print("✅ Saved")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--scanpath", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--captions-file", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000,
                    help="How many images to sample; -1 = use all")
    parser.add_argument("--seed", type=int,  default=42,
                    help="Random seed for reproducible sampling")
    parser.add_argument("--type", type=str, default="None") 
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--target-layer", type=int, default=None)
    args = parser.parse_args()
    main(args)
