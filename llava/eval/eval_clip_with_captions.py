from torchmetrics.multimodal.clip_score import CLIPScore
from PIL import Image
import json
import sys

def main() :
    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14-336")
    images_path = sys.argv[1]
    captions_path = sys.argv[2]

    
    captions_file = open(captions_path, "r") 
    captions_dict = json.load(captions_file)

    scores_per_image = []
    for id, captions in captions_dict :
        image_file = images_path + "/" + str(id) + ".jpg"
        image = Image.open(image_file)

        scores = []
        for caption in captions :
            score = metric(image, caption)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
    
        scores_per_image.append(avg_score)
    
    total_avg_score = sum(scores_per_image) / len(scores_per_image)

    print(f"Average CLIP Score : {total_avg_score}")


if __name__ == '__main__' :
    main()
