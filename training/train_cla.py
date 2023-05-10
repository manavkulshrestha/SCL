import torch
import torch.nn.functional as F

from Datasets.dutility import get_objdataloaders
from utility import save_model, device
from nn.Network import ObjectNet


def loss_fn(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


def train_epoch(model, epoch, loader, optimizer):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(loader.dataset)


def test_epoch(model, epoch, loader):
    # IMPORTANT, modified to only test predictions one batch at a time
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            actu = data.y.item()+1
            pred = model.prediction(data)
        correct += actu == pred

    return correct / len(loader.dataset)


def main():
    train_loader, val_loader, test_loader = get_objdataloaders()

    model = ObjectNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    best_val_acc = 0
    for epoch in range(1, 201):
        # get train loss and val acc
        train_loss = train_epoch(model, epoch, train_loader, optimizer)
        val_acc = test_epoch(model, epoch, test_loader)

        print(f'[Epoch {epoch:03d}] Train Loss: {train_loss:.4E}, Val Acc: {val_acc}')

        # save best model based on validation f1
        if best_val_acc <= val_acc:
            best_val_acc = val_acc
            save_model(model, 'cn_test_best_model.pt')

        # save a model every 20 epochs
        if epoch % 20 == 0:
            save_model(model, f'cn_test_model-{epoch}.pt')


if __name__ == '__main__':
    main()
