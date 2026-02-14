import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score

from training.dataset import CNNDailyMailExtractiveDataset
from model.extractive_model import ExtractiveSummarizer
from training.config import VOCAB_SIZE, PAD_IDX
from torch.utils.data import random_split



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4


def collate_fn(batch):
    input_ids, labels = zip(*batch)

    padded_inputs = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=PAD_IDX
    )

    labels = torch.stack(labels)

    return padded_inputs, labels


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    all_preds = []
    all_labels = []

    for batch_idx, (input_ids, labels) in enumerate(dataloader):

        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (outputs > 0.5).float()

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return total_loss / len(dataloader), acc, f1

def evaluate(model, dataloader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in dataloader:

            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids)

            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return acc, f1

def main():
    print("Loading dataset...")
    full_dataset = CNNDailyMailExtractiveDataset(
        split="train",
        max_samples=10000
    )

    # 90% train, 10% validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Initializing model...")
    model = ExtractiveSummarizer(vocab_size=VOCAB_SIZE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    print("Starting training...")

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1} =====")

        avg_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_acc, val_f1 = evaluate(model, val_loader)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        torch.save(
            model.state_dict(),
            f"checkpoints/extractive_epoch{epoch+1}.pth"
        )


if __name__ == "__main__":
    main()
