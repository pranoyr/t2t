
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
from bharat_kernel_pytorch.dataloader import QwenDummyDataset, dummy_data

def main():
    # 1. Initialize Accelerator
    # gradient_accumulation_steps helps simulate larger batch sizes on limited VRAM
    accelerator = Accelerator(gradient_accumulation_steps=4)
    
    # 2. Configuration
    model_id = "Qwen/Qwen2.5-Coder-7B" 
    dataset_name = "smangrul/code-chat-assistant-v1" # Example coding instruction dataset
    batch_size = 2
    learning_rate = 2e-5
    num_epochs = 3
    max_seq_length = 1024
    
    accelerator.print(f"Loading tokenizer and model: {model_id}")
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Qwen typically doesn't have a pad token by default; set it to eos_token for batched training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 4. Load Model
    # We leave device_map as None so Accelerate can distribute the model across GPUs automatically
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 # bf16 is highly recommended for modern LLMs like Qwen
    )
    
    # Enable gradient checkpointing to save massive amounts of VRAM
    model.gradient_checkpointing_enable()

    # 5. Load and Prepare Dataset
    accelerator.print("Loading custom dummy SFT dataset...")
    
    dummy_dataset = QwenDummyDataset(
        data=dummy_data, 
        tokenizer=tokenizer, 
        max_seq_length=max_seq_length
    )
    
    # DataCollator automatically batches inputs and preserves our target-masked labels
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    dataloader = DataLoader(
        dummy_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )

    
    # 6. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_training_steps = (len(dataloader) * num_epochs) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=total_training_steps
    )
    
    # 7. Prepare everything with Accelerate
    # This automatically wraps the model for DDP/FSDP and moves tensors to the correct GPUs
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # 8. The Training Loop
    accelerator.print("Starting training loop...")
    progress_bar = tqdm(range(total_training_steps), disable=not accelerator.is_local_main_process)
    
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Optimizer step (Accelerate handles the accumulation logic here)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # Update progress bar only when accumulation step completes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

    # 9. Save the Final Model
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