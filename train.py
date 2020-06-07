import time
import datetime
import random
import numpy as np
import pandas as pd
import sys
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
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


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


def main():
    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 512
    batch_size = 4
    epochs = 4
    learning_rate = 1e-6
    seed = 42
    model_name = "roberta-large"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Read data
    train = pd.read_csv("dataset/labeledTrainData.tsv", delimiter="\t")
    imdb_reviews = train["review"]
    sentiments = train["sentiment"]
    # Tokenize the text
    tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)

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
    train_size = int(0.90 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )

    valid_dataloader = DataLoader(
        valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=batch_size
    )

    # Load the model
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2, output_attentions=False, output_hidden_states=False
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_training_steps=total_steps, num_warmup_steps=0
    )
    train_stats = []
    total_time = time.time()

    # Finetune the model
    for epoch in range(0, epochs):
        print(f"========== Epoch {epoch + 1} ==========")
        # Train
        now = time.time()
        epoch_training_loss = 0
        model.train()
        for step, batch in enumerate(
            tqdm(train_dataloader, position=0, file=sys.stdout, leave=True)
        ):
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
        print("\nTrain loss: {0:.4f}".format(avg_train_loss))
        print("Training time: {:}\n".format(train_time))
        now = time.time()

        # Validate
        model.eval()
        epoch_eval_acc = 0
        epochs_eval_loss = 0
        eval_steps = 0
        for batch in tqdm(valid_dataloader, position=0, file=sys.stdout, leave=True):
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)
            with torch.no_grad():
                loss, logits = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    labels=batch_labels,
                )
            epochs_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            labels_ids = batch_labels.to("cpu").numpy()
            epoch_eval_acc += flat_accuracy(logits, labels_ids)

        avg_valid_acc = epoch_eval_acc / len(valid_dataloader)
        avg_valid_loss = epochs_eval_loss / len(valid_dataloader)
        print()
        print("Validation accuracy: {0:.4f}".format(avg_valid_acc))
        valid_time = format_time(time.time() - now)
        print("Validation loss: {0:.4f}".format(avg_valid_loss))
        print("Validation time: {:}".format(valid_time))

        train_stats.append(
            {
                "Epoch": epoch + 1,
                "Train loss": avg_train_loss,
                "Valid loss": avg_valid_loss,
                "Valid acc": avg_valid_acc,
                "Train time": train_time,
                "Validation time": valid_time,
            }
        )
        print(
            "Total training took {:} (hh:mm:ss)".format(
                format_time(time.time() - total_time)
            )
        )
        print()

    # Test
    test = pd.read_csv("dataset/testData.tsv", delimiter="\t")
    print("Number of test sentences: {:,}\n".format(test.shape[0]))
    reviews = test["review"]
    input_ids = []
    attention_masks = []

    for r in reviews:
        encoded_dict = tokenizer.encode_plus(
            r,
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
    labels = torch.tensor(labels)

    batch_size = 128

    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size
    )

    # Predict test data
    model.eval()
    predictions, true_labels = [], []
    for batch in tqdm(prediction_dataloader, position=0, file=sys.stdout, leave=True):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    # Make submission file
    pred = []
    for i in predictions:
        pred += list(np.argmax(i, axis=1))

    data = {"id": test["id"], "sentiment": pred}
    submission = pd.DataFrame(data)
    submission.to_csv("submission.csv", index=False)

    # Save the model and its tokenier
    # model.save_pretrained("/content/drive/My Drive/roberta-large-imdb")
    # tokenizer.save_pretrained("/content/drive/My Drive/roberta-large-imdb")


if __name__ == "__main__":
    main()
