# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging

import Config_mos as config
from Load_Dataset import ImageToImage2D
from utils import read_text, WeightedDiceBCE, iou_on_batch, dice_on_batch, save_on_batch
from local_segment_anything import sam_model_registry
from augmentation import get_augmentation_gray
import cv2

################################################################################
# Logger
################################################################################
def logger_config():
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    return logger


################################################################################
# Load Model
################################################################################
def load_model():
    config_sam = config.get_sam_config()

    model = sam_model_registry[config_sam.model_type](
        checkpoint=config_sam.checkpoint,
        adapter_flag=config_sam.adapter,
        interaction_indexes=config_sam.interaction_indexes,
        adapter_num_heads=config_sam.adapter_num_heads,
        downsample_rate=config_sam.downsample_rate,
        cff_ratio=config_sam.cff_ratio,
        text_cross=config_sam.text_cross,
        adapter_type=config_sam.adapter_type,
        attn_type=config_sam.attn_type,
        mlp_vit=config_sam.mlp_vit,
    )

    model = model.cuda()

    # 🔥 PUT YOUR TRAINED SESSION NAME HERE
    trained_session = "test_02.22_13h10"

    best_model_path = os.path.join(
        "BUSI_80-20_text",
        "sam",
        trained_session,
        "models",
        f"best_model-{config.model_name}.pth.tar"
    )
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    print(f"✅ Model loaded from: {best_model_path}")

    return model
################################################################################
# Metric computation
################################################################################
def compute_metrics(pred, mask):
    """
    pred: (B,1,H,W)
    mask: (B,1,H,W)
    """

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    mask = (mask > 0.5).float()

    TP = (pred * mask).sum()
    TN = ((1 - pred) * (1 - mask)).sum()
    FP = (pred * (1 - mask)).sum()
    FN = ((1 - pred) * mask).sum()

    smooth = 1e-6

    sensitivity = TP / (TP + FN + smooth)
    specificity = TN / (TN + FP + smooth)
    accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)
    dice = (2 * TP) / (2 * TP + FP + FN + smooth)
    iou = TP / (TP + FP + FN + smooth)

    return (
        sensitivity.item(),
        specificity.item(),
        accuracy.item(),
        dice.item(),
        iou.item()
    )

################################################################################
# Load Test Dataset
################################################################################
def load_test_dataset():
    test_tf = get_augmentation_gray([config.img_size, config.img_size], train_flag=False)

    test_text = read_text(config.text_val)

    test_dataset = ImageToImage2D(
        config.val_dataset,
        test_text,
        config.task_name,
        test_tf,
        image_size=config.img_size,
        mean_text_flag=config.mean_text_flag
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # print("")
    print(f"Test dataset size: {len(test_dataset)}")

    return test_loader


################################################################################
# Test Loop
################################################################################
def test():
    logger = logger_config()
    model = load_model()
    test_loader = load_test_dataset()

    sens_sum = 0
    spec_sum = 0
    acc_sum = 0
    dice_sum = 0
    iou_sum = 0

    save_path = os.path.join(config.save_path, "test_predictions")
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", ncols=120)

        for i, (sampled_batch, names) in enumerate(pbar):

            images = sampled_batch["image"].cuda()
            masks = sampled_batch["label"].cuda()
            text = sampled_batch["text"].cuda()

            preds = model(images, text)

            sens, spec, acc, dice, iou = compute_metrics(preds, masks)

            sens_sum += sens
            spec_sum += spec
            acc_sum += acc
            dice_sum += dice
            iou_sum += iou

            # 🔥 Update tqdm live metrics
            pbar.set_postfix({
                "Dice": f"{dice_sum/(i+1):.4f}",
                "IoU": f"{iou_sum/(i+1):.4f}"
            })

            # Save prediction
            pred_prob = torch.sigmoid(preds)
            pred_mask = (pred_prob > 0.5).float()

            pred_np = pred_mask[0, 0].cpu().numpy() * 255
            pred_np = pred_np.astype(np.uint8)

            save_file = os.path.join(save_path, names[0] + "_pred.png")
            cv2.imwrite(save_file, pred_np)

    n = len(test_loader)

    print("\n================ Final Test Results ================")
    print(f"Sensitivity (Recall): {sens_sum/n:.4f}")
    print(f"Specificity         : {spec_sum/n:.4f}")
    print(f"Accuracy            : {acc_sum/n:.4f}")
    print(f"IoU (Jaccard)       : {iou_sum/n:.4f}")
    print(f"Dice (F1-score)     : {dice_sum/n:.4f}")
    print("====================================================")

################################################################################
# Run
################################################################################
if __name__ == "__main__":
    test()