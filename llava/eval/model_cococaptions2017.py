import argparse
import torch
import os
import json
import numpy as np
from collections import defaultdict

from PIL import Image
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, process_scanpaths, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from transformers import TextStreamer

BATCH_SIZE = 8

def load_image(image_file) :
    image = Image.open(image_file).convert('RGB')
    return image

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
    
    # Caption prompt

    questions = [
        DEFAULT_IMAGE_TOKEN + '\n' + "Describe the image concisely in one sentence, up to 20 words only.",
        DEFAULT_IMAGE_TOKEN + '\n' + "Render a clear and concise summary of the photo in one sentence, up to 20 words only.",
        DEFAULT_IMAGE_TOKEN + '\n' + "Share a concise interpretation of the image provided in one sentence, up to 20 words only.",
        DEFAULT_IMAGE_TOKEN + '\n' + "Give a brief description of the image in one sentence, up to 20 words only.",
    ]

    # questions = [
    #     DEFAULT_IMAGE_TOKEN + '\n' + "Describe the image 
    # ]

    # Ground truth captions file
    captions_file = json.load(open(os.path.expanduser(args.captions_file), "r"))
    
    # Response
    output_captions = defaultdict(list)
    
    image_ids = list(captions_file.keys())

    for question in questions :
        for i in range(0, len(image_ids), BATCH_SIZE):
    
            batch_keys = image_ids[i : i + BATCH_SIZE]
            prompts = []
            all_images = []
            all_image_ids = []
            all_scanpaths = []
            image_sizes = []

            for image_id in batch_keys : 
                img_path = os.path.join(images_dir, image_id + ".jpg")
                scan_path = os.path.join(scanpaths_dir, image_id + "_scanpath.npy")

                image = load_image(img_path)
                scanpath = [np.load(scan_path, allow_pickle=True).item()]

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)  
                prompts.append(conv.get_prompt())
                all_images.append(image)
                all_scanpaths.append(scanpath)
                all_image_ids.append(image_id)
                image_sizes.append(image.size)
            
            image_tensor, new_scanpaths = process_images(all_images, image_processor, model.config, all_scanpaths)
            processed_scanpaths = process_scanpaths(new_scanpaths)
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
                        # output_attentions=True,
                        return_dict_in_generate=True,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True)
            outputs = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)
            for j in range(len(outputs)) :
                cur_id = all_image_ids[j]
                cur_output = outputs[j]
                output_captions[cur_id].append(cur_output)
            print(f"Done batch {i // BATCH_SIZE}!", flush=True)
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    json.dump(dict(output_captions), ans_file, indent=2)

    return

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
    
    args = parser.parse_args()
    main(args)