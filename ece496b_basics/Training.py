#!/usr/bin/env python3
import argparse
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ece496b_basics import Transformer_LM, lr_schedule, checkpoint, Data_loading, gradient_clipping, cross_entropy, AdamW



def train(args):
    # Initialize TensorBoard logging
    writer = SummaryWriter(log_dir="runs/Tuned_lr3e-3_32batch_owt_nominlr")
    start_time = time.time()
    
    #Load training and validation data with memory mapping
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")
    
    # Initialize model and optimizer
    model = Transformer_LM.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        weights={}  # Use random initialization if no pre-trained weights provided
    ).to(args.device)
    
    optimizer = AdamW.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    #resume from a checkpoint
    start_iteration = 0
    if args.resume_checkpoint:
        start_iteration = checkpoint.load_checkpoint(args.resume_checkpoint, model, optimizer)
        print(f"Loaded checkpoint with iteration {start_iteration}")
        if args.reset_iter:
            start_iteration = 0
            print("Reset iteration counter to 0 for fine tuning.")

    
    #training loop
    for iteration in range(start_iteration, (args.max_iters + 1)):
        model.train()
        inputs, targets = Data_loading.get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy.cross_entropy(logits.view(-1, args.vocab_size),
        targets.view(-1))
        loss.backward()
        gradient_clipping.gradient_clipping(model.parameters(), args.grad_clip_max)
        optimizer.step()
    
        #Learning rate schedule 
        new_lr = lr_schedule.cosine_learning_rate_schedule(
            t=iteration,
            alpha_max=args.lr,
            alpha_min=args.lr_min,
            T_w=args.warmup_iters,
            T_c=args.anneal_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
        # logging of training progress
        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            writer.add_scalar("Train/Loss", loss.item(), iteration)
            writer.add_scalar("Train/LearningRate", new_lr, iteration)
            writer.add_scalar("Time/Elapsed", elapsed, iteration)
            print(f"Iteration {iteration}: Train Loss = {loss.item():.4f} | LR = {new_lr:.6f}")
    
        #save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            checkpoint.save_checkpoint(model, optimizer, iteration, args.checkpoint_path)
            print(f"Checkpoint saved at iteration {iteration}")
    
        # Periodically evaluate on validation set
        if iteration % args.val_interval == 0 and iteration > 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = Data_loading.get_batch(val_data, args.batch_size, args.context_length, args.device)
                val_logits = model(val_inputs)
                val_loss = cross_entropy.cross_entropy(val_logits.view(-1, args.vocab_size),
                                                        val_targets.view(-1))
            writer.add_scalar("Validation/Loss", val_loss.item(), iteration)
            writer.add_scalar("validation and Time", val_loss.item(), elapsed)
            print(f"Iteration {iteration}: Validation Loss = {val_loss.item():.4f}")
    
    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    
    # Data + model hyperparameters
    parser.add_argument("--train_data", type=str, required=True, help="Path to .npy training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to .npy validation data")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.15, help="Dropout in attention")
    parser.add_argument("--residual_pdrop", type=float, default=0.15, help="Dropout in residual connections")
    
    # Optimization hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-3, help="Base learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-4, help="Minimum LR for cosine schedule")
    parser.add_argument("--warmup_iters", type=int, default= 100, help="Steps of warmup")
    parser.add_argument("--anneal_iters", type=int, default= 2100, help="Steps until min LR is reached")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip_max", type=float, default=1.3)
    
    # Loop settings
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--val_interval", type=int, default=50)
    
    # Checkpointing
    parser.add_argument("--checkpoint_path", type=str, default="owt_trained.pt")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--reset_iter", action="store_true",help="Reset iteration counter to 0 when resuming from a checkpoint.")

    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
