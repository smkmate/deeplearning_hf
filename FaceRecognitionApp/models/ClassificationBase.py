import torch
import torch.nn as nn


# function for calculating accuracy
def accuracy(preds, labels):
    # not binary
    if preds.shape[1] != 1:
        _, preds = torch.max(preds, dim=1)
        _, labels = torch.max(labels, dim=1)
    # binary
    else:
        preds = torch.round(preds)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    # training step
    def training_step(self, batch):
        img, targets = batch            # images and their labels
        out = self(img)                 # passing the image through the model
        targets = targets.type(torch.float32)

        # setting the activation
        if out.shape[1] == 1:
            loss_fn = nn.BCELoss()          # for binary
        else:
            loss_fn = nn.CrossEntropyLoss() # for multi-class

        loss = loss_fn(out, targets)
        return loss

    # validation step
    def validation_step(self, batch):
        # getting the prediction
        img, targets = batch
        targets = targets.type(torch.float32)
        out = self(img)

        # setting the activation
        if out.shape[1] == 1:
            loss_fn = nn.BCELoss()  # for binary
        else:
            loss_fn = nn.CrossEntropyLoss()  # for multi-class

        loss = loss_fn(out, targets)
        targets = targets.type(torch.int32)
        # accuracy
        acc = accuracy(out, targets)
        return {'val_acc': acc, 'val_loss': loss}

    # validation step at the end of the epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # Printing the results at the end of the epoch
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
              .format(epoch, result["train_loss"], result["val_loss"], result["val_acc"]))
