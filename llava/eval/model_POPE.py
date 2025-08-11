import argparse
import torch
import os
import json
import jsonlines
import sys
import numpy as np
import gc
import matplotlib.pyplot as plt
import wget
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Rectangle
from torchvision import transforms
from contextlib import contextmanager

NSERC = Path(__file__).resolve().parents[3]  # .../NSERC
HAT   = NSERC / "HAT"
sys.path.insert(0, str(HAT))
sys.path.insert(0, str(NSERC))

BATCH_SIZE = 8

from HAT.common.config import JsonConfig
from HAT.hat.models import HumanAttnTransformer
from HAT.hat.evaluation import scanpath_decode

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, process_scanpaths, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

def plot_scanpath(img, xs, ys, bbox=None, title=None, save_dir=None):
    fig, ax = plt.subplots()
    ax.imshow(img)
    cir_rad = 15

    dpi = 100
    fig.set_dpi(dpi)
    fig.set_size_inches(img.width / dpi, img.height / dpi)

    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                      ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)

    for i in range(len(xs)):
        
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            edgecolor='red',
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(
            i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")

    if bbox is not None:
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
            alpha=0.5, edgecolor='yellow', facecolor='none', linewidth=2)
        ax.add_patch(rect)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    plt.show()
    plt.savefig(save_dir, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

@contextmanager
def chdir(path: Path) :
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)

def load_image(image_file):
    return Image.open(image_file).convert('RGB')

def prepare_mask_type_dict(type_=None, margin=0, target_layer=None):
    mask_type = {}
    mask_type["type"] = "None" if type_ is None else type_
    mask_type["margin"] = margin
    if target_layer is not None:
        mask_type["target-layer"] = target_layer
    return mask_type

def save_helper(answers_path, output_captions):
    with jsonlines.open(answers_path, mode="a") as writer:
        writer.write(output_captions)
    print("✅ Saved")

def actions2scanpaths(norm_fixs, im_h, im_w):
    # convert actions to scanpaths
    scanpaths = []
    for fixs in norm_fixs:
        fixs = fixs.numpy()
        scanpaths.append({
            'X': fixs[:, 0] * im_w,
            'Y': fixs[:, 1] * im_h,
        })
    return scanpaths

def main(args):
    disable_torch_init()
    
    images_dir = Path(args.images_dir).expanduser()
    answers_path = Path(args.answers_file).expanduser()
    answers_path.parent.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("CUDA device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "None", flush=True)

    ## Load / Setup LLaVA
    
    model_name = get_model_name_from_path(args.model_path)
    
    tokenizer, model_LLAVA, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        args.load_8bit, args.load_4bit, device=args.device
    )

    print("Model device:", model_LLAVA.device, flush=True)

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
        print(f"[WARNING] Auto-inferred conversation mode is {conv_mode}, "
              f"but `--conv-mode` is {args.conv_mode}, using {args.conv_mode}")
        conv_mode = args.conv_mode
    args.conv_mode = conv_mode

    ## Load / Setup HAT
    with chdir(HAT):
        TAP = 'FV'
        hparams = JsonConfig(str(HAT / "configs" / "coco_freeview_dense_SSL.json"))

        if not os.path.exists(f"./checkpoints/HAT_{TAP}.pt"):
            if not os.path.exists("./checkpoints/"):
                os.mkdir('./checkpoints')

            print('downloading model checkpoint...')
            url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/HAT_{TAP}.pt"
            wget.download(url, 'checkpoints/')

        if not os.path.exists(f"./pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl"):
            if not os.path.exists("./pretrained_models/"):
                os.mkdir('./pretrained_models')

            print('downloading pretrained model weights...')
            url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl"
            wget.download(url, 'pretrained_models/')
            url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50.pkl"
            wget.download(url, 'pretrained_models/')

        # create model
        model_HAT = HumanAttnTransformer(
            hparams.Data,
            num_decoder_layers=hparams.Model.n_dec_layers,
            hidden_dim=hparams.Model.embedding_dim,
            nhead=hparams.Model.n_heads,
            ntask=1 if hparams.Data.TAP == 'FV' else 18,
            tgt_vocab_size=hparams.Data.patch_count + len(hparams.Data.special_symbols),
            num_output_layers=hparams.Model.num_output_layers,
            separate_fix_arch=hparams.Model.separate_fix_arch,
            train_encoder=hparams.Train.train_backbone,
            train_pixel_decoder=hparams.Train.train_pixel_decoder,
            use_dino=hparams.Train.use_dino_pretrained_model,
            dropout=hparams.Train.dropout,
            dim_feedforward=hparams.Model.hidden_dim,
            parallel_arch=hparams.Model.parallel_arch,
            dorsal_source=hparams.Model.dorsal_source,
            num_encoder_layers=hparams.Model.n_enc_layers,
            output_centermap="centermap_pred" in hparams.Train.losses,
            output_saliency="saliency_pred" in hparams.Train.losses,
            output_target_map="target_map_pred" in hparams.Train.losses,
            transfer_learning_setting=hparams.Train.transfer_learn,
            project_queries=hparams.Train.project_queries,
            is_pretraining=False,
            output_feature_map_name=hparams.Model.output_feature_map_name)
        
        checkpoint_paths = {
            'TP': "./checkpoints/HAT_TP.pt", # target present
            'TA': "./checkpoints/HAT_TA.pt", # target absent
            'FV': "./checkpoints/HAT_FV.pt" # free viewing
        }

        ckpt = torch.load(checkpoint_paths[hparams.Data.TAP], map_location='cpu')
    
    bb_weights = ckpt['model']
    bb_weights_new = bb_weights.copy()
    for k, v in bb_weights.items() :
        if "stages." in k :
            new_k = k.replace("stages.", "")
            bb_weights_new[new_k] = v 
            bb_weights_new.pop(k)

    model_HAT.load_state_dict(bb_weights_new)

    
    questions = Path(args.questions_file).expanduser()
    questions = list(jsonlines.open(questions))

    mask_type = prepare_mask_type_dict(type_=args.type, margin=args.margin, target_layer=args.target_layer)


    size = (hparams.Data.im_h, hparams.Data.im_w)
    transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    sample_action = False
    task_id = torch.tensor([0], dtype=torch.long)

    for batch_start in tqdm(
        range(0, len(questions), BATCH_SIZE),
    ):
        batch_images = questions[batch_start : batch_start + BATCH_SIZE]
        prompts, scanpaths, image_sizes, all_images, texts = (
            [],
            [],
            [],
            [],
            []
        )

        with torch.inference_mode(): 
            for img_question in batch_images :
                question_id, image, text = img_question['question_id'], img_question['image'], img_question['text']        
                texts.append(text)
                image = images_dir / image

                conv = conv_templates[conv_mode].copy()
                prompt_body = DEFAULT_IMAGE_TOKEN + "\n" + text
                conv.append_message(conv.roles[0], prompt_body)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())

                pil_img = load_image(image)
                image_sizes.append(pil_img.size)
                all_images.append(pil_img)
                original_resolution = pil_img.size
                X_ratio = original_resolution[0] / 512 
                Y_ratio = original_resolution[1] / 320         

                img_tensor = torch.unsqueeze(transform(pil_img), 0)
                normalized_sp, _ = scanpath_decode(model_HAT, img_tensor, task_id, hparams.Data, sample_action=sample_action, center_initial=True)
                scanpath = actions2scanpaths(normalized_sp, hparams.Data.im_h, hparams.Data.im_w)[0]

                scanpath['X'] = scanpath['X'] * X_ratio
                scanpath['Y'] = scanpath['Y'] * Y_ratio
                scanpaths.append([scanpath])

        try :
            image_tensor, new_scanpaths = process_images(all_images, image_processor, model_LLAVA.config, scanpaths)
            processed_scanpaths = process_scanpaths(new_scanpaths, mode=args.trajectory)

            if isinstance(image_tensor, list):
                image_tensor = [img.to(model_LLAVA.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model_LLAVA.device, dtype=torch.float16)

            tokenized = [
                tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
                for prompt in prompts
            ]

            # Pad to max length
            max_len = max([t.size(0) for t in tokenized])
            padded = [
                torch.cat([t, torch.full((max_len - t.size(0),), tokenizer.pad_token_id, dtype=torch.long)])
                for t in tokenized
            ]

            input_ids = torch.stack(padded, dim=0).to(model_LLAVA.device)
            
            # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in prompts], dim=0).to(model_LLAVA.device)
            with torch.inference_mode():
                output_ids = model_LLAVA.generate(
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
                    use_cache=True
                )

                outputs = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)
            
            for i in range(len(batch_images)) :
                save_helper(answers_path, {"question" : texts[i], "answer" : outputs[i]})

        finally:
            for var in [
                'weights', 'image_infos', 'data', 'outputs', 'output_ids',
                'image_tensor', 'input_ids', 'processed_scanpaths',
                'image_sizes', 'prompts', 'all_images', 'all_scanpaths'
            ]:
                if var in locals():
                    del locals()[var]

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
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
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--questions-file", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000, help="How many images to sample; -1 = use all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--type", type=str, default="None")
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--target-layer", type=int, default=None)
    args = parser.parse_args()
    main(args)