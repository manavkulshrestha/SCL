import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from Datasets.BaseDatasets import MLPDataset
from nn.Baselines import BaseMLP
from utility import save_model


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0

    for x, y in dataloader:
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def test_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.view(-1)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return running_loss / len(dataloader), (correct / total) * 100


def main():
    torch.manual_seed(142)

    lr = 0.0000008
    model = BaseMLP(10, 511).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = MLPDataset('../Datasets/data/baselinedata_791_0-10-1684965474', chunk=(0, 632), flat=True)
    test_dataset = MLPDataset('../Datasets/data/baselinedata_791_0-10-1684965474', chunk=(632, 791), flat=True)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Training loop
    num_epochs = 200

    best_model_acc = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy = test_epoch(model, test_loader, criterion)

        if best_model_acc <= test_accuracy and test_accuracy > 74:
            best_model_acc = test_accuracy
            save_model(model, f'mlp_baseline_{int(best_model_acc)}.pt')

        print(f"Epoch [{epoch+1:03d}/{num_epochs}] - {lr=}"
              f" - Train Loss: {train_loss:.4f}"
              f" - Test Loss: {test_loss:.4f}"
              f" - Test Accuracy: {test_accuracy:.2f}%")

    print("Training finished.")


if __name__ == "__main__":
    main()
