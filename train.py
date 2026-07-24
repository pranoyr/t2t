
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen Coder on GPUMODE/KernelBook SFT dataset")
    parser.add_argument("--model_id", type=str, default="Qwen2.5-Coder-7B-Instruct", help="Pretrained model ID")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
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
def evaluate(model, eval_dataloader, accelerator, max_eval_steps=20):
    model.eval()
    total_loss = 0.0
    num_steps = 0
    for step, batch in enumerate(eval_dataloader):
        if step >= max_eval_steps:
            break
        outputs = model(**batch)
        loss = outputs.loss
        gathered_loss = accelerator.gather_for_metrics(loss)
        total_loss += gathered_loss.mean().item()
        num_steps += 1
    model.train()
    return total_loss / max(1, num_steps)


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
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
    model.gradient_checkpointing_enable()

    accelerator.print("Loading GPUMODE/KernelBook SFT datasets (train & eval)...")
    train_dataloader = get_kernelbook_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        # max_seq_length=args.max_seq_length,
        split=args.train_split
    )
    eval_dataloader = get_kernelbook_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        # max_seq_length=args.max_seq_length,
        split=args.eval_split
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_training_steps = (len(train_dataloader) * args.num_epochs) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=total_training_steps
    )
    
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_train_samples = len(train_dataloader.dataset)
    num_eval_samples = len(eval_dataloader.dataset)
    steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps

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
    
    global_step = 0
    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1} - Step {global_step} - Loss: {loss.item():.4f}")
                
                # Logging
                if global_step % args.log_interval == 0 and accelerator.is_main_process and wandb is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch + 1,
                        "train/step": global_step,
                    })
                
                # Running evals
                if global_step % args.eval_interval == 0:
                    accelerator.print(f"\nRunning evaluation at step {global_step}...")
                    eval_loss = evaluate(model, eval_dataloader, accelerator, max_eval_steps=args.max_eval_steps)
                    accelerator.print(f"Step {global_step} - Eval Loss: {eval_loss:.4f}")
                    if accelerator.is_main_process and wandb is not None:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "train/step": global_step,
                        })
                
                # Saving intermediate checkpoints
                if global_step % args.save_interval == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        accelerator.print(f"Saving intermediate checkpoint to {checkpoint_dir}...")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            checkpoint_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save
                        )
                        tokenizer.save_pretrained(checkpoint_dir)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"Training complete! Saving final model to {args.output_dir}...")
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        tokenizer.save_pretrained(args.output_dir)
        accelerator.print(f"Model saved to {args.output_dir}")
        if wandb is not None:
            wandb.finish()

if __name__ == "__main__":
    main()