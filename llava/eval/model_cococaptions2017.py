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
    
def question_image_attention_viewer(image_ids, data) :
    attn_weights, image_infos_all = data['attn_weights'], data['image_infos']
    for idx, id in enumerate(image_ids) :
        img_path = os.path.expanduser(f"~/NSERC/samples/may26_samples/sampled_images_1000/{id}.jpg")
        image = Image.open(img_path)
        image = expand2square(image, background_color=(0, 0, 0))

        image_infos = image_infos_all[idx]
        
        head_idx = 0

        for layer_idx in range(len(attn_weights)) :
            attn = attn_weights[layer_idx][idx][head_idx]
            sq_len = attn.shape[1]
            
            image_start = image_infos['start_index']
            image_len = image_infos['num_patches']
            image_end = image_start + image_len

            image_indices = list(range(image_start, image_end))

            question_span_1 = list(range(0, image_start))
            question_span_2 = list(range(image_end, sq_len))
            question_indices = question_span_1 + question_span_2

            attn_q_to_img = attn[question_indices][:, image_indices]
            avg_attn = attn_q_to_img.mean(dim=0)
            
            side = int(image_len ** 0.5)
            assert side * side == image_len 
            # —————— PLOT & SAVE THE RAW PATCH‐LEVEL HEATMAP ——————
            attn_map = avg_attn.reshape(side, side).cpu().numpy()  # shape (side, side)

            base_save_dir = "/scratch/ssd004/scratch/merc0606/qToImg"
            save_dir = os.path.join(base_save_dir, f"attention_maps_{id}")
            os.makedirs(save_dir, exist_ok=True)

            # —————— OVERLAY ON THE FULL IMAGE ——————
            # 1. Determine the “patch_size” (in pixels) used by your model: here it's 14.
            patch_size = 14

            # 2. Resize the square image so that its size = (side * patch_size, side * patch_size).
            full_size = side * patch_size  # e.g. 24 * 14 = 336
            image_resized = image.resize((full_size, full_size), resample=Image.BILINEAR)

            # 3. Normalize attn_map to [0,1]
            attn_min, attn_max = attn_map.min(), attn_map.max()
            if (attn_max - attn_min) > 1e-6:
                attn_norm = (attn_map - attn_min) / (attn_max - attn_min)
            else:
                attn_norm = attn_map * 0.0  # if it’s constant, just all zeros

            # 4. Upsample from (side, side) → (full_size, full_size) using bicubic
            attn_pil = Image.fromarray((attn_norm * 255).astype(np.uint8))  # single‐channel
            attn_upsampled = attn_pil.resize((full_size, full_size), resample=Image.BICUBIC)
            attn_upsampled = np.array(attn_upsampled) / 255.0              # back to float [0,1]

            # 5. Convert to an RGB colormap (e.g. “jet”)
            cmap = plt.get_cmap("jet")
            heatmap_rgb = cmap(attn_upsampled)[:, :, :3]  # shape (H, W, 3), values in [0,1]

            # 6. Blend the original image with the heatmap (alpha=0.5)
            orig_np = np.array(image_resized).astype(np.float32) / 255.0      # shape (H, W, 3), [0,1]
            overlay_np = orig_np * 0.5 + heatmap_rgb * 0.5                    # simple 50/50 mix
            overlay_img = Image.fromarray((overlay_np * 255).astype(np.uint8))

            # 7. Save the blended overlay
            overlay_img.save(os.path.join(save_dir, f"overlay_on_image_layer_{layer_idx}.png"))

            print("✅ Saved",flush=True)

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
    
    if args.weights_dir is not None :
        weights_dir = args.weights_dir
        os.makedirs(weights_dir, exist_ok=True)
    if args.trajectory is not None :
        mode = args.trajectory
    print(f"✅ Running Job, with parameters \n scanpaths : {scanpaths_dir} \n images : {images_dir} \n weights : {weights_dir}")

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

    for question_num, question in enumerate(questions) :
        total_batches = (len(image_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in tqdm(range(0, len(image_ids), BATCH_SIZE), total=total_batches, ascii=True, file=sys.stdout):
    
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
                        output_attentions=True,
                        return_dict_in_generate=True,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True)
            

            outputs = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)
            
            # Code for trying to save attention heat maps for image-question attention
            if question_num == 0 and i < 8 :
                weights = output_ids['attentions']
                image_infos = output_ids['image_infos']
                data = {
                    'attn_weights': weights,
                    'image_infos': image_infos,
                    'all_image_ids': all_image_ids,
                }
                torch.save(data, os.path.join(weights_dir, f"{i // BATCH_SIZE}_weights.pt"))
                print(f"Saved weights to {weights_dir} ✅", flush=True)
                print(f"Sample outputs : {outputs}")

            # question_image_attention_viewer(all_image_ids,data)
            for j in range(len(outputs)) :
                cur_id = all_image_ids[j]
                cur_output = outputs[j]
                output_captions[cur_id].append(cur_output)
    
            # print(f"Done batch {i // BATCH_SIZE}!", flush=True)
    
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
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--trajectory", type=int, default=0)
    args = parser.parse_args()
    main(args)