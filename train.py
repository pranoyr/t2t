
import os
import argparse
import torch
try:
    import wandb
except ImportError:
    wandb = None
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
from tqdm.auto import tqdm
from bharat_kernel_pytorch.dataloader import get_kernelbook_dataloader

def save_custom_checkpoint(accelerator, model, optimizer, lr_scheduler, tokenizer, global_step, checkpoint_dir):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    
    unwrapped_model.save_pretrained(
        checkpoint_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True
    )
    
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
        training_state = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "global_step": global_step
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

def load_custom_checkpoint(accelerator, model, optimizer, lr_scheduler, checkpoint_dir):
    accelerator.print(f"Loading custom checkpoint from {checkpoint_dir}")
    
    unwrapped_model = accelerator.unwrap_model(model)
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        unwrapped_model.load_state_dict(state_dict)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        unwrapped_model.load_state_dict(state_dict)
    else:
        accelerator.print(f"Warning: Could not find model weights in {checkpoint_dir}")

    training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        optimizer.load_state_dict(training_state["optimizer"])
        lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        return training_state.get("global_step", 0)
    return 0

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen Coder on GPUMODE/KernelBook SFT dataset")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Pretrained model ID")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--rewrite_weights", action=argparse.BooleanOptionalAction, default=True, help="Overwrite the latest checkpoint instead of creating new ones")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="./qwen-coder-finetuned", help="Directory to save checkpoints")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    # Interval arguments (num_iters for logging, saving, and evals)
    parser.add_argument("--log_interval", type=int, default=10, help="Number of optimizer steps between wandb logs")
    parser.add_argument("--eval_interval", type=int, default=100, help="Number of optimizer steps between evals")
    parser.add_argument("--save_interval", type=int, default=500, help="Number of optimizer steps between saving checkpoints")
    parser.add_argument("--max_eval_steps", type=int, default=20, help="Max eval batches per evaluation run")
    parser.add_argument("--train_split", type=str, default="train[:98%]", help="Dataset split for training")
    parser.add_argument("--eval_split", type=str, default="train[98%:]", help="Dataset split for evaluation")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, eval_dataloader, accelerator, tokenizer, max_eval_steps=20):
    model.eval()
    total_loss = 0.0
    num_steps = 0
    
    eval_generations = []
    
    for step, batch in enumerate(eval_dataloader):
        if step >= max_eval_steps:
            break
        outputs = model(**batch)
        loss = outputs.loss
        gathered_loss = accelerator.gather_for_metrics(loss)
        total_loss += gathered_loss.mean().item()
        num_steps += 1
        
        # Generate for the first 5 samples
        if accelerator.is_main_process and len(eval_generations) < 5:
            try:
                input_ids_batch = batch["input_ids"]
                labels_batch = batch["labels"]
                
                for i in range(input_ids_batch.size(0)):
                    if len(eval_generations) >= 5:
                        break
                        
                    input_ids = input_ids_batch[i:i+1]
                    labels = labels_batch[i]
                    
                    valid_label_indices = (labels != -100).nonzero()
                    if len(valid_label_indices) > 0:
                        prompt_len = valid_label_indices[0].item()
                        prompt_ids = input_ids[:, :prompt_len]
                        
                        eval_prompt = tokenizer.decode(prompt_ids[0])
                        
                        unwrapped = accelerator.unwrap_model(model)
                        gen_ids = unwrapped.generate(
                            prompt_ids,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.2,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True
                        )
                        eval_generated = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=False)
                        eval_generations.append([eval_prompt, eval_generated])
            except Exception as e:
                accelerator.print(f"Warning: Evaluation generation failed: {e}")
                
    model.train()
    return total_loss / max(1, num_steps), eval_generations


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16"
    )
    
    if accelerator.is_main_process and wandb is not None:
        wandb.init(
            project="bharat-kernel",
            name="qwen2.5-coder-7b-sft",
            config=vars(args)
        )
    elif accelerator.is_main_process and wandb is None:
        accelerator.print("Warning: wandb not installed. Skipping W&B logging.")
    
    accelerator.print(f"Loading tokenizer and model: {args.model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    accelerator.print("Loading GPUMODE/KernelBook SFT datasets (train & eval)...")
    train_dataloader = get_kernelbook_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        split=args.train_split
    )
    eval_dataloader = get_kernelbook_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        split=args.eval_split
    )
    
    # Try to use 8-bit AdamW to save ~42GB of VRAM
    try:
        import bitsandbytes as bnb
        accelerator.print("Using bitsandbytes 8-bit AdamW optimizer (Saving ~42GB VRAM!)")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    except ImportError:
        accelerator.print("bitsandbytes not found. Falling back to standard PyTorch AdamW (High VRAM usage).")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    import math
    steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * args.num_epochs
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=total_training_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)
    
    global_step = 0
    starting_epoch = 0
    resume_step = 0
    if args.resume_from_checkpoint:
        global_step = load_custom_checkpoint(accelerator, model, optimizer, lr_scheduler, args.resume_from_checkpoint)
        if global_step > 0:
            starting_epoch = (global_step * accelerator.gradient_accumulation_steps) // len(train_dataloader)
            resume_step = (global_step * accelerator.gradient_accumulation_steps) % len(train_dataloader)
            accelerator.print(f"Resumed global step: {global_step}, starting epoch: {starting_epoch}, resuming step: {resume_step}")
    
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_train_samples = len(train_dataloader.dataset)
    num_eval_samples = len(eval_dataloader.dataset)

    accelerator.print("=" * 60)
    accelerator.print("                 TRAINING CONFIGURATION SUMMARY             ")
    accelerator.print("=" * 60)
    accelerator.print(f"  Model ID                      : {args.model_id}")
    accelerator.print(f"  Training samples              : {num_train_samples:,}")
    accelerator.print(f"  Evaluation samples            : {num_eval_samples:,}")
    accelerator.print(f"  Number of Epochs              : {args.num_epochs}")
    accelerator.print(f"  Per-device Batch Size         : {args.batch_size}")
    accelerator.print(f"  Number of Devices (GPUs)      : {accelerator.num_processes}")
    accelerator.print(f"  Gradient Accumulation Steps   : {args.gradient_accumulation_steps}")
    accelerator.print(f"  Effective Global Batch Size   : {total_batch_size}")
    accelerator.print(f"  Iterations (Steps) per Epoch  : {steps_per_epoch:,}")
    accelerator.print(f"  Total Training Iterations     : {total_training_steps:,}")
    accelerator.print("=" * 60)
    accelerator.print("Starting training loop...\n")
    progress_bar = tqdm(range(total_training_steps), disable=not accelerator.is_local_main_process)

    if global_step > 0:
        progress_bar.update(global_step)
    
    model.train()
    for epoch in range(starting_epoch, args.num_epochs):
        active_dataloader = train_dataloader
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step > 0:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                if torch.isnan(loss):
                    accelerator.print(f"\n--- FATAL: NaN LOSS DETECTED ---")
                    accelerator.print(f"Step: {step}, Global Step: {global_step}")
                    accelerator.print(f"Model weight dtype: {next(model.parameters()).dtype}")
                    accelerator.print(f"Input IDs max: {batch['input_ids'].max()}, min: {batch['input_ids'].min()}")
                    accelerator.print(f"Labels max: {batch['labels'].max()}, min: {batch['labels'].min()}")
                    
                    if hasattr(outputs, "logits") and outputs.logits is not None:
                        accelerator.print(f"Logits contains NaN: {torch.isnan(outputs.logits).any().item()}")
                        accelerator.print(f"Logits contains Inf: {torch.isinf(outputs.logits).any().item()}")
                    import sys; sys.exit(1)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # CRITICAL FIX: PyTorch SDPA + Padding tokens often produces NaN gradients for the padding tokens
                    # in the backward pass. If we don't zero them out, clip_grad_norm_ will divide by NaN, 
                    # immediately corrupting every single weight in the model on the very first optimizer step.
                    for p in model.parameters():
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                            
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1} - Step {global_step} - Loss: {loss.item():.4f}")
                
                # Logging
                if global_step % args.log_interval == 0 and accelerator.is_main_process and wandb is not None:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + 1,
                        "train/step": global_step,
                    })
                
                # Running evals
                if global_step % args.eval_interval == 0:
                    accelerator.print(f"\nRunning evaluation at step {global_step}...")
                    eval_loss, eval_generations = evaluate(model, eval_dataloader, accelerator, tokenizer, max_eval_steps=args.max_eval_steps)
                    accelerator.print(f"Step {global_step} - Eval Loss: {eval_loss:.4f}")
                    if accelerator.is_main_process and wandb is not None:
                        wandb_log_dict = {
                            "eval/loss": eval_loss,
                            "train/step": global_step,
                        }
                        if eval_generations:
                            table = wandb.Table(columns=["Prompt", "Generated Code"], data=eval_generations)
                            wandb_log_dict["eval/generations"] = table
                            
                        wandb.log(wandb_log_dict)
                
                # Saving intermediate checkpoints
                if global_step % args.save_interval == 0:
                    accelerator.wait_for_everyone()
                    if args.rewrite_weights:
                        checkpoint_dir = os.path.join(args.output_dir, "checkpoint-latest")
                    else:
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.print(f"Saving intermediate checkpoint to {checkpoint_dir}...")
                    save_custom_checkpoint(accelerator, model, optimizer, lr_scheduler, tokenizer, global_step, checkpoint_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"Training complete! Saving final model to {args.output_dir}...")
        
    save_custom_checkpoint(accelerator, model, optimizer, lr_scheduler, tokenizer, global_step, args.output_dir)
    if accelerator.is_main_process:
        accelerator.print(f"Final Model saved to {args.output_dir}")
        if wandb is not None:
            wandb.finish()

if __name__ == "__main__":
    main()