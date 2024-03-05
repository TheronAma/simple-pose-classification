import torch
import argparse
import wandb
import numpy as np
from tqdm import tqdm
from data.dataloader import PoseDataset
from data.constants import ACTIONS
from model.pose_classifier import PoseClassifier

config = {
        "epochs" : 50,
        "hidden_size" : 256,
        "batch_size" : 256,
        "init_lr" : 0.01,
        "momentum" : 0.9,
        "weight_decay" : 1e-4
        }

def eval(model, dataloader, criterion):

    model.eval()
    vloss, vacc =  0, 0
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="val")

    for i, (poses, actions) in enumerate(dataloader):

        poses.to(device)
        actions.to(device)

        with torch.no_grad():
            logits = model(poses)
            loss = criterion(logits, actions)

        vloss += loss.item()
        vacc  += torch.sum(torch.argmax(logits, dim=1) == actions).item() / (logits.shape[0])

        batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))),
                              acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
        batch_bar.update()

        del actions, poses, logits

        torch.cuda.empty_cache()

    batch_bar.close()
    vloss   /= len(val_loader)
    vacc    /= len(val_loader)

    return vloss, vacc

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0, 0
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="train")

    for i, (pose, actions) in enumerate(dataloader):
        optimizer.zero_grad()

        pose.to(device)
        actions.to(device)

        logits = model(pose)

        loss = criterion(logits, actions)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_acc  += torch.sum(torch.argmax(logits, dim=1) == actions).item() / (logits.shape[0])

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i+1))),
            acc="{:.04f}%".format(float(total_acc * 100 / (i + 1))))
        batch_bar.update()

        """
        if i == 1:
            print("Pose: ", pose[0])
            print("Actions: ", actions)
            print("Logits: ", logits)
        """
        del pose, actions, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss /= len(train_loader)
    total_acc /= len(train_loader)

    return total_loss, total_acc

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", default="datasets/jackrabbot")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)

    wandb_run = wandb.init(
        name="init-model-small-split",
        reinit=True,
        project="pose-classification",
        config=config
    )

    train_data = PoseDataset(args.dataset, actions=ACTIONS)

    val_data = PoseDataset(args.dataset, actions=ACTIONS, partition="val")

    model = PoseClassifier(hidden_size=config["hidden_size"])

    train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        num_workers = 0,
        batch_size = config["batch_size"],
        pin_memory = True,
        shuffle = True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset = val_data,
        num_workers = 0,
        batch_size = config["batch_size"],
        pin_memory = True,
        shuffle = False
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["init_lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=1e-4, factor=0.75, mode="max")

    for epoch in range(config["epochs"]):
        tloss, tacc = train(model, train_loader, optimizer, criterion)
        vloss, vacc = eval(model, val_loader, criterion)

        scheduler.step(vacc)

        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(vacc, vloss))

        wandb.log({"train_loss":tloss, 'train_Acc': tacc, 'validation_Acc':vacc,
                   'validation_loss': vloss, "learning_Rate": float(optimizer.param_groups[0]['lr'])})

