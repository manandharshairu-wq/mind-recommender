import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.model import NRMSModel
from src.evaluate import evaluate


def train_model(train_loader, val_loader, embedding_matrix, device, epochs=5):

    import os
    os.makedirs("models", exist_ok=True)   # 👈 FIX HERE


    model = NRMSModel(embedding_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in loop:
            history = batch['history'].to(device)
            candidates = batch['candidates'].to(device)
            labels = batch['labels'].to(device)
            hist_mask = batch['hist_mask'].to(device)

            optimizer.zero_grad()

            scores = model(history, candidates, hist_mask)

            # convert one-hot labels → class index
            target = torch.argmax(labels, dim=1)

            loss = criterion(scores, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        print(f"\nEpoch {epoch+1}: Loss = {avg_loss:.4f}")

        print("Running evaluation...")
        evaluate(model, val_loader, device)

        torch.save(model.state_dict(), f"models/nrms_epoch{epoch+1}.pt")

    return model, losses