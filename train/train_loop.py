import torch
from tqdm import tqdm
import torch.nn as nn
import os
from ..train.losses import captioning_loss
from ..utils.param_count import count_active_parameters
from ..config import RESUME_CHECKPOINT_PATH, CHECKPOINT_SAVE_DIR, vocab_size
def train_model(model, train_loader, val_loader, tokenizer, device='cuda', epochs=20):
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0

    if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT_PATH}")
        checkpoint = torch.load(RESUME_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from the end of Epoch {start_epoch-1}. Starting Epoch {start_epoch}.")
    else:
        print("No resume checkpoint found. Starting training from scratch.")

    print(f"Training on {device}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_task_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            video = batch['video'].to(device)
            input_ids = batch['input_ids'].to(device)
            targets = input_ids[:, 1:]
            
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda',):
                # --- This is the full, correct forward and loss calculation ---
                logits, diagnostics = model(video, input_ids[:, :-1])
                
                task_loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                
                total_aux_loss = 0.0
                # --- FIX: Iterate over .items(), not .item() ---
                if isinstance(diagnostics, dict):
                    for key, value in diagnostics.items():
                        if 'loss' in key:
                            total_aux_loss += value

                alpha = 0.01 
                total_loss_batch = task_loss + alpha * total_aux_loss

            # The full, correct AMP backward pass
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- FIX: Correctly accumulate the task loss ---
            epoch_task_loss += task_loss.item()
            # --- FIX: Cleaned up progress bar to show both losses ---
            pbar.set_postfix({'TaskLoss': f'{task_loss.item():.4f}', 'AuxLoss': f'{total_aux_loss:.4f}'})

        avg_loss = epoch_task_loss / len(train_loader)
        
        # We need a forward pass to update router stats for an accurate count
        model.eval()
        with torch.no_grad():
            _ = model(video, input_ids[:, :-1])
        active_params, total_params, active_by_layer = count_active_parameters(model)
        
        print(f"\nEpoch {epoch+1} finished. Avg Task Loss: {avg_loss:.4f}, Perplexity: {torch.exp(torch.tensor(avg_loss)):.2f}")
        print(f"  Params -> Active: {active_params/1e6:.2f}M | Total: {total_params/1e6:.2f}M | Ratio: {active_params/total_params:.2%}")
        for layer_name, experts in active_by_layer.items():
            print(f"    - Layer '{layer_name}' active experts: {experts}")

        # Checkpointing logic
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
        }
        latest_path = os.path.join(CHECKPOINT_SAVE_DIR, 'latest_checkpoint.pth')
        torch.save(checkpoint_data, latest_path)
        print(f"✓ Checkpoint saved to {latest_path}\n")

    return model

# =========================================================================