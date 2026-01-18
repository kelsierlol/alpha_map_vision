import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import logging

from vision.models.unet_model import UNet2D
from vision.models.alpha_model import AlphaController2D, local_redundancy
from vision.data_utils.mvtec_dataset import MVTecADDataset
from vision.data_utils.corruptions import corrupt_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    # Ensure scores and labels are 1D arrays
    scores = scores.flatten()
    labels = labels.flatten()

    # Filter out NaN scores and corresponding labels
    valid_indices = ~np.isnan(scores)
    scores = scores[valid_indices]
    labels = labels[valid_indices]

    if len(np.unique(labels)) < 2:
        # PR-AUC is not well-defined if there's only one class
        return 0.0

    order = np.argsort(-scores)
    labels_sorted = labels[order]
    
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    
    # Avoid division by zero
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(labels_sorted.sum(), 1e-12)
    
    # Add (0,0) and (1,0) points to PR curve
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [1.0]])

    trap = getattr(np, "trapz", np.trapz) # Use np.trapz for older numpy versions
    return trap(precision, recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha map anomaly detection on MVTec AD dataset.")
    parser.add_argument("--data-dir", type=str, default="data/mvtec_ad", help="Root directory for MVTec AD dataset.")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category to use (e.g., 'bottle').")
    parser.add_argument("--img-size", type=int, default=256, help="Image size for training and evaluation.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for data loaders.")
    parser.add_argument("--epochs-unet", type=int, default=10, help="Number of epochs to train the UNet.")
    parser.add_argument("--epochs-alpha", type=int, default=10, help="Number of epochs to train the Alpha Controller.")
    parser.add_argument("--lr-unet", type=float, default=1e-4, help="Learning rate for UNet.")
    parser.add_argument("--lr-alpha", type=float, default=5e-5, help="Learning rate for Alpha Controller.")
    parser.add_argument("--redundancy-window", type=int, default=5, help="Window size for local redundancy calculation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--plot", action="store_true", help="Generate and save example plots.")
    parser.add_argument("--output-dir", type=str, default="outputs/mvtec_alpha_demo", help="Directory to save plots and models.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup Data Loaders
    logging.info(f"Loading MVTec AD '{args.category}' dataset...")
    train_dataset = MVTecADDataset(
        root_dir=args.data_dir, category=args.category, is_train=True, resize=args.img_size
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = MVTecADDataset(
        root_dir=args.data_dir, category=args.category, is_train=False, resize=args.img_size
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    logging.info("Dataset loading complete.")

    # 2. Initialize Models
    model_unet = UNet2D(in_ch=3, base=64).to(device) # Using the larger UNet2D
    controller_alpha = AlphaController2D().to(device)
    logging.info("Models initialized.")

    # 3. Stage 1: Train UNet reconstruction
    logging.info("--- Stage 1: Training UNet for reconstruction ---")
    opt_unet = torch.optim.Adam(model_unet.parameters(), lr=args.lr_unet)
    model_unet.train()
    for epoch in range(1, args.epochs_unet + 1):
        total_loss = 0.0
        for i, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            recon = model_unet(x)
            loss = F.l1_loss(recon, x) # L1 loss is good for reconstruction
            
            opt_unet.zero_grad(set_to_none=True)
            loss.backward()
            opt_unet.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0: # Log every 10 batches
                logging.info(f"UNet Epoch {epoch:02d}/{args.epochs_unet} | Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"UNet Epoch {epoch:02d}/{args.epochs_unet} | Average Loss: {avg_loss:.6f}")
    
    # Save trained UNet
    unet_path = os.path.join(args.output_dir, "unet_model.pth")
    torch.save(model_unet.state_dict(), unet_path)
    logging.info(f"UNet model saved to {unet_path}")

    # 4. Stage 2: Train Alpha Controller on known corruptions
    logging.info("--- Stage 2: Training Alpha Controller ---")
    opt_alpha = torch.optim.Adam(controller_alpha.parameters(), lr=args.lr_alpha)
    controller_alpha.train()
    model_unet.eval() # UNet is fixed during alpha controller training
    rng = torch.Generator(device=device).manual_seed(args.seed + 1)

    for epoch in range(1, args.epochs_alpha + 1):
        total_loss = 0.0
        for i, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            
            # Apply synthetic corruptions to the original image
            corrupt, mask = corrupt_batch(x, rng, 
                                        occlusion_size=args.img_size // 8, 
                                        blur_size=args.img_size // 10,
                                        sp_size=args.img_size // 12,
                                        copy_size=args.img_size // 8)
            
            with torch.no_grad():
                recon = model_unet(corrupt)
            
            # Calculate residual magnitude and local redundancy
            resid_mag = (recon - corrupt).abs().mean(dim=1, keepdim=True).detach()
            
            # Convert corrupt to grayscale before calculating redundancy
            grayscale_corrupt = transforms.Grayscale()(corrupt)
            redundancy = local_redundancy(grayscale_corrupt, args.redundancy_window).detach()
            
            # Predict alpha map
            alpha = controller_alpha(resid_mag, redundancy)

            # Convert alpha to a score for binary classification (corruption vs. no corruption)
            # This score indicates the likelihood of a pixel being corrupted
            # Original uses sigmoid(-(alpha - alpha0) / 0.2)
            # Let's try to make it more direct for BCE loss with mask
            # We want alpha to be high for corrupted regions, low for uncorrupted
            # So, a simple approach could be to directly use alpha scaled to [0,1]
            score = torch.sigmoid(alpha) # Sigmoid to scale alpha to a probability-like score

            loss = F.binary_cross_entropy(score, mask) # Compare with ground truth corruption mask
            
            opt_alpha.zero_grad(set_to_none=True)
            loss.backward()
            opt_alpha.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0: # Log every 10 batches
                logging.info(f"Alpha Epoch {epoch:02d}/{args.epochs_alpha} | Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Alpha Controller Epoch {epoch:02d}/{args.epochs_alpha} | Average Loss: {avg_loss:.6f}")

    # Save trained Alpha Controller
    controller_path = os.path.join(args.output_dir, "alpha_controller.pth")
    torch.save(controller_alpha.state_dict(), controller_path)
    logging.info(f"Alpha Controller model saved to {controller_path}")

    # 5. Quick Evaluation and Plotting
    logging.info("--- Evaluation ---")
    controller_alpha.eval()
    model_unet.eval()
    
    all_scores_flat = []
    all_labels_flat = []
    
    with torch.no_grad():
        for i, (x, gt_mask, label_img_level) in enumerate(test_loader):
            x = x.to(device)
            # No synthetic corruption here; we test on original test images
            # and detect anomalies that are *inherent* in the test set.
            # The model learns to find 'corruptions' that deviate from normalcy.
            
            # For demonstration, we can apply *some* corruption here to show how alpha works
            # but for actual anomaly detection, we evaluate on *original* test data
            # Let's use the original `x` as `corrupt` for evaluation to find inherent anomalies
            # or apply minimal corruption for visualization
            
            # For a proper MVTec evaluation, we assume 'x' contains anomalies
            # and our model should detect them. The 'mask' will be the GT anomaly mask.
            
            # Reconstruct the potentially anomalous image
            recon = model_unet(x)
            
            # Calculate residual magnitude and local redundancy
            resid_mag = (recon - x).abs().mean(dim=1, keepdim=True)
            redundancy = local_redundancy(x, args.redundancy_window) # Calculate redundancy on original test image
            
            # Predict alpha map
            alpha = controller_alpha(resid_mag, redundancy)
            
            # Using alpha as anomaly score (higher alpha implies more anomalous)
            # The exact score mapping might need tuning for best performance
            anomaly_score_map = torch.sigmoid(alpha) # Scale to [0,1] for interpretability
            
            # Store scores and labels for PR-AUC calculation
            all_scores_flat.append(anomaly_score_map.cpu().numpy().flatten())
            all_labels_flat.append(gt_mask.cpu().numpy().flatten()) # Using MVTec's GT masks

            if args.plot and i < 5: # Plot first 5 test samples
                fig, axs = plt.subplots(1, 4, figsize=(12, 3))
                
                # Original Image
                img_display = x[0].permute(1, 2, 0).cpu().numpy()
                # Un-normalize image for display
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_display = std * img_display + mean
                img_display = np.clip(img_display, 0, 1)

                axs[0].imshow(img_display)
                axs[0].set_title(f"Original (Anomaly: {label_img_level[0].item()})")
                axs[0].axis("off")

                # Alpha map
                heat = anomaly_score_map[0, 0].cpu().numpy()
                axs[1].imshow(heat, cmap="viridis")
                axs[1].set_title("Alpha Score Map")
                axs[1].axis("off")

                # Predicted Anomaly Mask (simple thresholding for visualization)
                pred_mask = (heat > 0.5).astype(np.float32) # Simple threshold
                axs[2].imshow(pred_mask, cmap="Reds")
                axs[2].set_title("Predicted Mask")
                axs[2].axis("off")

                # Ground Truth Mask
                gt_m = gt_mask[0, 0].cpu().numpy()
                axs[3].imshow(gt_m, cmap="Reds")
                axs[3].set_title("Ground Truth Mask")
                axs[3].axis("off")
                
                fig.tight_layout()
                plot_path = os.path.join(args.output_dir, f"mvtec_alpha_demo_sample_{i}.png")
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                logging.info(f"Saved plot for sample {i} to {plot_path}")

    # Calculate overall PR-AUC
    all_scores_flat = np.concatenate(all_scores_flat)
    all_labels_flat = np.concatenate(all_labels_flat)
    pixel_pr_auc = pr_auc(all_scores_flat, all_labels_flat)
    logging.info(f"Pixel-level PR-AUC on MVTec test set: {pixel_pr_auc:.4f}")

    if args.plot:
        # Example of how the alpha map and corruption mask relate using a synthetic corruption
        logging.info("Generating synthetic corruption example plot...")
        x_example, _, _ = next(iter(test_loader))
        x_example = x_example[:1].to(device) # Take one sample
        
        corrupt_example, mask_example = corrupt_batch(x_example, rng, 
                                                occlusion_size=args.img_size // 8, 
                                                blur_size=args.img_size // 10)
        
        with torch.no_grad():
            recon_example = model_unet(corrupt_example)
            resid_example = (recon_example - corrupt_example).abs().mean(dim=1, keepdim=True)
            redundancy_example = local_redundancy(corrupt_example, args.redundancy_window)
            alpha_example = controller_alpha(resid_example, redundancy_example)
            alpha_score_example = torch.sigmoid(alpha_example)

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        
        img_display = corrupt_example[0].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)

        axs[0].imshow(img_display)
        axs[0].set_title("Synthetically Corrupted")
        axs[0].axis("off")

        axs[1].imshow(alpha_score_example[0, 0].cpu().numpy(), cmap="viridis")
        axs[1].set_title("Alpha Score Map")
        axs[1].axis("off")

        axs[2].imshow(mask_example[0, 0].cpu().numpy(), cmap="Reds")
        axs[2].set_title("Ground Truth Corruption Mask")
        axs[2].axis("off")
        
        fig.tight_layout()
        plot_path_synthetic = os.path.join(args.output_dir, "mvtec_synthetic_corruption_demo.png")
        fig.savefig(plot_path_synthetic, dpi=150)
        plt.close(fig)
        logging.info(f"Saved synthetic corruption demo to {plot_path_synthetic}")


if __name__ == "__main__":
    # Temporarily set MPLCONFIGDIR to a writable directory in case it's not set
    # This is often needed in environments where default home directory is not writable
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
    main()
