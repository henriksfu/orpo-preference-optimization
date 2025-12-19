import gc
import os
import json
import gzip
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import ORPOTrainer, ORPOConfig
from peft import LoraConfig

def get_data(path):
    save_path = path.replace('.gz', '')
    if not os.path.exists(save_path):
        print('Unzipping input files ...')
        with gzip.open(path, 'rt', encoding='utf-8') as f_in:
            with open(save_path, 'wt', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
    else:
        print('Input files are already unzipped.')
        pass
    with open(save_path, 'rt', encoding='utf-8') as f:
        return [line.strip() for line in f]

def preprocess_data():
    prompt_raw = get_data('data/train.txt.gz')
    chosen_raw = get_data('data/train.out.gz')
    rejected_raw = get_data('data/train_default.out.gz')

    prompts = []
    chosens = []
    rejects = []

    for prompt, chosen, rejected in zip(prompt_raw, chosen_raw, rejected_raw):
        prompt_data = json.loads(prompt)
        chosen_data = json.loads(chosen)
        rejected_data = json.loads(rejected)
        prompts.append([
            {
                "role": "system", 
                "content": "You are a helpful assistant that provides useful answers without too much extra output."
            },
            {
                "role": "user", 
                "content": prompt_data['prompt'] + '\n' + prompt_data['constraints']
            }
        ])
        chosens.append([
            {
                "role": "assistant", 
                "content": chosen_data['output']
            }
        ])
        rejects.append([
            {
                "role": "assistant", 
                "content": rejected_data['output']
            }
        ])

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejects
    })

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = "./model"
    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = preprocess_data()
    dataset = dataset.shuffle(seed=42).select(range(8000))
    dataset = dataset.train_test_split(test_size=0.01)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
    )

    orpo_config = ORPOConfig(
        output_dir=model_dir,
        beta=0.1,
        max_steps=1000,
        max_length=512,
        max_prompt_length=256,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=8e-6,
        logging_steps=50,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        eval_strategy="no",
        lr_scheduler_type="cosine"
    )

    quantization_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_cfg,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  

    trainer = ORPOTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,   
        args=orpo_config,
        peft_config=lora_config
    )
    trainer.train()
    trainer.save_model(model_dir)
    print("Training complete and model saved:", model_dir)

if __name__ == "__main__":
    main()
