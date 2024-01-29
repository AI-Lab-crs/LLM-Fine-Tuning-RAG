import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer,AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer,TrainingArguments
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    return [line.strip() for line in data]

# def split_dataset(input_sequences, target_sequences, labels, split_ratio=[0.8, 0.1, 0.1]):
#     # Concatenate input and target sequences for easier splitting
#     sequences = list(zip(input_sequences, target_sequences, labels))

#     # Calculate split sizes based on split_ratio
#     split_sizes = [int(ratio * len(sequences)) for ratio in split_ratio]
#     split_sizes[-1] = len(sequences) - sum(split_sizes[:-1])

#     # Split the sequences into train, validation, and test sets
#     train_sequences = sequences[:split_sizes[0]]
#     val_sequences = sequences[split_sizes[0]:split_sizes[0] + split_sizes[1]]
#     test_sequences = sequences[split_sizes[0] + split_sizes[1]:]

#     # Unpack the sequences into separate lists
#     train_inputs, train_targets, train_labels = zip(*train_sequences)
#     val_inputs, val_targets, val_labels = zip(*val_sequences)
#     test_inputs, test_targets, test_labels = zip(*test_sequences)

#     # Return the split datasets
#     return (
#         list(train_inputs), list(train_targets), list(train_labels),
#         list(val_inputs), list(val_targets), list(val_labels),
#         list(test_inputs), list(test_targets), list(test_labels)
#     )


# def generate_text(input_sequences, model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    inputs = tokenizer.batch_encode_plus(
        input_sequences,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,  # Maximum length of generated text
        num_return_sequences=1  # Number of text sequences to generate
    )

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_text
def load_base_model(base_model):
    model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config(),
    device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def quant_config():
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    return quant_config

def split_dtype(path):
    dataset = load_dataset(path, split="train")
    return dataset

def tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def parameter_efficient_Fine_tuning():
    peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    )
    return peft_params

def set_training_params():
    training_params = TrainingArguments(
    output_dir="../results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
    )
    return training_params

def fine_tune(base_model,dataset,peft_params, tokenizer,training_params):
    trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
    )
    return trainer