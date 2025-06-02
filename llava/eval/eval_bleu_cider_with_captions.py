import os 
import json
import sys
import torch
from torchmetrics.text import BLEUScore

def main() :
    generated_captions_path = sys.argv[1]
    ground_truth_captions_path = sys.argv[2]
    save_path = sys.argv[3]

    with open(ground_truth_captions_path) as f:
        gt = json.load(f)
    
    with open(generated_captions_path,"r") as f:
        target = json.load(f)

    BLEU_scores = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bleu = BLEUScore(smooth=True).to(device)
    image_ids = target.keys()
    sum_ = 0
    for image_id in image_ids :
        gt_captions = gt[image_id]
        generated_captions = target[image_id]
            
        indiv_scores = []
        for sample in generated_captions :
            score = bleu([sample], [gt_captions])
            indiv_scores.append(score.item())
        sum_ += max(indiv_scores)
        BLEU_scores[image_id] = {
            "scores" : indiv_scores,
            "avg_score" : sum(indiv_scores) / len(indiv_scores)
        }

    BLEU_scores['overall_avg'] = {
        "average" : sum_ / len(image_ids)
    }
    # breakpoint()
    with open(save_path, "w") as f :
        json.dump(BLEU_scores, f, indent=2)

if __name__ == '__main__' :
    main()