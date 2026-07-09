
import torch
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

def main():
    accelerator = Accelerator(gradient_accumulation_steps=4)
    
    model_id = "Qwen/Qwen2.5-Coder-7B"
    batch_size = 2
    learning_rate = 2e-5
    num_epochs = 3
    max_seq_length = 1024
    
    accelerator.print(f"Loading tokenizer and model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()

    accelerator.print("Loading GPUMODE/KernelBook SFT dataset...")
    dataloader = get_kernelbook_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        split="train"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_training_steps = (len(dataloader) * num_epochs) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=total_training_steps
    )
    
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    accelerator.print("Starting training loop...")
    progress_bar = tqdm(range(total_training_steps), disable=not accelerator.is_local_main_process)
    
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_main_process:
        accelerator.print("Training complete! Saving model...")
        unwrapped_model.save_pretrained(
            "./qwen-coder-finetuned",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        tokenizer.save_pretrained("./qwen-coder-finetuned")
        accelerator.print("Model saved to ./qwen-coder-finetuned")

if __name__ == "__main__":
    main()