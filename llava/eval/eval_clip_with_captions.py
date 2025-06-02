from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import sys
import os
import torch 
import gc

def main() :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14-336").to(device)
    # metric = CLIPScore(model_name_or_path="zer0int/LongCLIP-L-Diffusers")
    # metric.processor.image_processor.do_rescale = False

    images_path = sys.argv[1]
    captions_path = sys.argv[2]
    save_path = sys.argv[3]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(captions_path, "r") as f:
        captions_dict = json.load(f)

    max_scores_per_image = []
    all_scores = {}

    for id in captions_dict.keys() :
        captions = captions_dict[id]
        image_file = os.path.join(images_path, str(id) + ".jpg")

        image = Image.open(image_file).convert("RGB")
        image_tensor = transforms.functional.pil_to_tensor(image).to(device)

        scores = []
        with torch.no_grad() :
            for caption in captions :
                score = metric(image_tensor, caption)
                scores.append(score)


        max_score = float(max(scores))
        max_scores_per_image.append(max_score)
        all_scores[id] = {
            'max' : max_score,
            'all' : [float(s) for s in scores]
        }
        
        del image_tensor, scores
        gc.collect()

        print(f"Score for {id} computed!",flush=True)

    avg_max_score = sum(max_scores_per_image) / len(max_scores_per_image)
    all_scores['avg_score'] = avg_max_score


    with open(f"{save_path}.json", "w") as f:
        json.dump(all_scores, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.hist(max_scores_per_image, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Max CLIP Scores per Image")
    plt.xlabel("CLIP Score")
    plt.ylabel("Number of Images")
    plt.grid(True)
    plt.savefig(f"{save_path}.png")

if __name__ == '__main__' :
    main()
