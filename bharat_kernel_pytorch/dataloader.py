import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, default_data_collator, DataCollatorForSeq2Seq

class KernelBookDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
 
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Prompt text only (system + user + assistant generation header)
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        do_truncate = self.max_seq_length is not None

        encodings = self.tokenizer(
            full_text,
            truncation=do_truncate,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
    
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=do_truncate,
            max_length=self.max_seq_length
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # CRITICAL SAFEGUARD: Never allow 100% of the sequence to be masked out.
        # This prevents PyTorch from throwing a NaN loss if a sample has an empty completion.
        if len(input_ids) > 0:
            prompt_len = min(prompt_len, len(input_ids) - 1)
    
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def get_kernelbook_dataloader(tokenizer, batch_size=2, max_seq_length=None, split="train"):
    """
    Downloads GPUMODE/KernelBook, formats it for ChatML, applies target masking, 
    and returns a ready-to-use PyTorch DataLoader with dynamic padding (no truncation by default).
    """
    print("Loading filtered local dataset...")
    raw_dataset = load_dataset("json", data_files="filtered_kernelbook.jsonl", split="train")
    
    def format_to_messages(example):
        # Gracefully handle varying column names in case of dataset updates
        raw_python_code = example.get("prompt", example.get("instruction", example.get("question", example.get("python_code", ""))))
        target_kernel = example.get("completion", example.get("output", example.get("answer", example.get("triton_code", ""))))
        
        # We append the exact verbose prompt the user uses during inference to perfectly align the model
        verbose_instruction = (
            "You are given a pytorch function, and your task is to write the same\n"
            "triton implementation for it.\n"
            "The triton implementation should change the name from Model to\n"
            "ModelNew, and have same input and output as the pytorch function.\n\n"
            "Optimize the architecture with custom Triton kernels! Name your\n"
            "optimized output architecture ModelNew. Output the new code in\n"
            "codeblocks. Please generate real code, NOT pseudocode, make sure the\n"
            "code compiles and is fully functional. Just output the new model\n"
            "code, no input and init function, no other text, and NO testing\n"
            "code! **Return ONLY Python code using `@triton.jit`. Remember to Name your optimized output architecture\n"
            "ModelNew, do not use Model again!**\n\n"
            "Now, you need to write the triton implementation for the following\n"
            "pytorch code:\n```python\n"
            f"{raw_python_code}\n```"
        )
        
        return {
            "messages": [
                {"role": "system", "content": "You are an expert GPU kernel developer specialized in writing highly optimized Triton kernels."},
                {"role": "user", "content": verbose_instruction},
                {"role": "assistant", "content": target_kernel}
            ]
        }
        
    print("Formatting dataset into ChatML structure...")
    formatted_dataset = raw_dataset.map(format_to_messages, remove_columns=raw_dataset.column_names)
    
    if max_seq_length is not None:
        print(f"Filtering out samples exceeding max_seq_length ({max_seq_length})...")
        def filter_long_samples(example):
            messages = example["messages"]
            
            # Remove samples that have an empty target completion (which causes 100% masking)
            if not any(m["role"] == "assistant" and m["content"].strip() != "" for m in messages):
                return False
                
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(full_text, truncation=False)
            return len(tokenized["input_ids"]) <= max_seq_length
            
        original_size = len(formatted_dataset)
        formatted_dataset = formatted_dataset.filter(filter_long_samples)
        print(f"Filtered dataset from {original_size} to {len(formatted_dataset)} samples.")
    
    print("Initializing PyTorch Dataset with target masking...")
    train_dataset = KernelBookDataset(
        data=formatted_dataset, 
        tokenizer=tokenizer, 
        max_seq_length=max_seq_length
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    
    return dataloader



if __name__ == "__main__":
    # Initialize the tokenizer
    model_id = "Qwen/Qwen2.5-Coder-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Get the dataloader
    train_dataloader = get_kernelbook_dataloader(
        tokenizer=tokenizer, 
        batch_size=1, 
        max_seq_length=None
    )
    
    # Test it by pulling one batch
    for batch in train_dataloader:
        # print("\n--- BATCH SHAPES ---")
        # print("Input IDs shape:", batch["input_ids"].shape)
        # print("Attention Mask shape:", batch["attention_mask"].shape)
        # print("Labels shape:", batch["labels"].shape)
        
        # Verify masking worked (should see -100 at the start of the labels)
        # print("\nFirst 50 label tokens (Notice the -100s for the prompt):")
        # print(batch["labels"][0].tolist())
        
        # Verify actual text content from KernelBook is present
        sample_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
        print(sample_text)
        break