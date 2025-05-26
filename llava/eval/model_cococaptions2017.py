import argparse
import torch
import os
import json
import numpy as np

from PIL import Image
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, process_scanpaths, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from transformers import TextStreamer

def load_image(image_file) :
    image = Image.open(image_file).convert('RGB')
    return image

def main(args) :
    disable_torch_init()

    # model preparation
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, 
                                                              args.model_base, 
                                                              model_name, 
                                                              args.load_8bit, 
                                                              args.load_4bit, 
                                                              device=args.device)
    

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
    question = "Describe the image concisely."
    # question = "Provide a caption for the given image"
    question = DEFAULT_IMAGE_TOKEN + '\n' + question

    # Ground truth captions file
    captions_file = json.load(open(os.path.expanduser(args.captions_file), "r"))
    
    # Response
    output_captions = {}

    for caption in captions_file :
        # Conversation Template : conv_llava_v1
        conv = conv_templates[args.conv_mode].copy()

        img_file_name = images_dir + "/" + caption + ".jpg"
        scanpath_file_name = scanpaths_dir + "/" + caption + "_scanpath.npy"
        
        scanpath = [np.load(scanpath_file_name, allow_pickle=True).item()]
        ground_truth_captions = captions_file[caption]

        image = load_image(img_file_name)
        image_size = image.size

        # resized image and rescaled scanpaths
        image_tensor, new_scanpaths = process_images([image], image_processor, model.config, scanpath)
        processed_scanpaths = process_scanpaths(new_scanpaths)

        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # prompt generation for model
        conv.append_message(conv.roles[0], question)
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
                scanpaths=processed_scanpaths,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

            outputs = tokenizer.decode(output_ids.sequences[0]).strip()
            # conv.messages[-1][-1] = outputs
        
        outputs = outputs.replace("<s> ","")
        outputs = outputs.replace("</s>","")
        
        output_captions[str(caption)] = outputs

    # File to write down model response
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    json.dump(output_captions, ans_file, indent=2)


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