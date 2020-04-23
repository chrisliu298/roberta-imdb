import time
import datetime
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaModel,
)
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 512
batch_size = 4
epochs = 3
learning_rate = 1e-5
seed = 42

# Settings
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def format_time(elapsed):
    """
    Returns time elapse in hh:mm:ss format
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    """
    Format accuracy
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Read data
df = pd.read_csv("labeledTrainData.tsv", delimiter="\t")
imdb_reviews = df["review"]
sentiments = df["sentiment"]

# Tokenize the text
tokenizer = RobertaTokenizer.from_pretrained("roberta-large", do_lower_case=True)

input_ids = []
attention_masks = []

for review in imdb_reviews:
    encoded_dict = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(sentiments)

# Make data loader
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(datset, split_ratio=[train_size, valid_size])

print("{:>5,} training samples".format(train_size))
print("{:>5,} validation samples".format(val_size))

train_dataloader = DataLoader(
    train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
)

valid_dataloader = DataLoader(
    valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=batch_size
)

# Import the model
model = RobertaForSequentialClassification.from_pretrained(
    "roberta-large", num_labels=2, output_attentions=False, output_hidden_states=False
)
model.to(device)

optimizer = AdamW(model.parameter(), le=learning_rate, ops=1e-8)

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_training_steps=total_steps, num_warmup_steps=0
)


# Finetune the model
train_stats = []
total_time = time.time()

for epoch in range(0, epochs):
    print("=" * 10 + f" Epoch {epoch}/{epochs} " + "=" * 10)

    # Train
    print("Training started ...")
    now = time.time()
    epoch_training_loss = 0
    model.train()  # train mode
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - now)
            print(
                "Batch {:>5,} of {:>5,}.  Elapsed: {:}.".format(
                    step, len(train_dataloader), elapsed
                )
            )
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        model.zero_grad()
        loss, logits = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_masks,
            labels=batch_labels,
        )
        epoch_training_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = epoch_training_loss / len(train_dataloader)
    train_time = format_time(time.time() - now)
    print("\nAverage train loss: {0:.2f}".format(avg_train_loss))
    print("Training epcoh took: {:}".format(train_time))

    # Validation
    print("\nValidation started ...")
    now = time.time()
    model.eval()  # evaluation mode
    epoch_eval_acc = 0
    epochs_eval_loss = 0
    eval_steps = 0

    for batch in valid_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        with torch.no_grad():
            loss, logits = model(
                input_ids=batch_input_ids,
                attention_masks=batch_attention_masks,
                labels=batch_labels,
            )
        epochs_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        labels_ids = batch_labels.to("cpu").numpy()
        epoch_eval_acc += flat_accuracy(logits, labels_ids)

    avg_valid_acc = epoch_eval_acc / len(valid_dataloader)
    avg_valid_loss = epochs_eval_loss / len(valid_dataloader)
    print("Accuracy: {0:.2f}".format(avg_valid_acc))

    valid_time = format_time(time.time() - now)
    print("Validation Loss: {0:.2f}".format(avg_valid_loss))
    print("Validation took: {:}".format(valid_time))

    train_stats.append(
        {
            "epoch": epoch + 1,
            "Train loss": avg_train_loss,
            "Valid loss": avg_valid_loss,
            "Valid acc": avg_valid_acc,
            "Train Time": train_time,
            "Validation Time": valid_time,
        }
    )

    print()
    print("Finetuning completed.")
    print(
        "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_time))
    )
