from transformers import GPT2LMHeadModel, GPT2Tokenizer
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    return [line.strip() for line in data]

def split_dataset(input_sequences, target_sequences, labels, split_ratio=[0.8, 0.1, 0.1]):
    # Concatenate input and target sequences for easier splitting
    sequences = list(zip(input_sequences, target_sequences, labels))

    # Calculate split sizes based on split_ratio
    split_sizes = [int(ratio * len(sequences)) for ratio in split_ratio]
    split_sizes[-1] = len(sequences) - sum(split_sizes[:-1])

    # Split the sequences into train, validation, and test sets
    train_sequences = sequences[:split_sizes[0]]
    val_sequences = sequences[split_sizes[0]:split_sizes[0] + split_sizes[1]]
    test_sequences = sequences[split_sizes[0] + split_sizes[1]:]

    # Unpack the sequences into separate lists
    train_inputs, train_targets, train_labels = zip(*train_sequences)
    val_inputs, val_targets, val_labels = zip(*val_sequences)
    test_inputs, test_targets, test_labels = zip(*test_sequences)

    # Return the split datasets
    return (
        list(train_inputs), list(train_targets), list(train_labels),
        list(val_inputs), list(val_targets), list(val_labels),
        list(test_inputs), list(test_targets), list(test_labels)
    )


def generate_text(input_sequences, model_name):
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

input_sequences = ['sfsd', 'dsdf']
model_name = "gpt2"  # Name or identifier of the pre-trained GPT-2 model

generatgenerated_texted_text = generate_text(input_sequences, model_name)
print(generated_text)