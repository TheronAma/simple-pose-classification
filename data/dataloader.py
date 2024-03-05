import torch
import numpy as np
from .constants import ACTIONS

class PoseDataset(torch.utils.data.Dataset):

    def __init__(self, root, actions, partition="train"):
        self.actions = actions

        self.data = np.load(f"{root}/{partition}/data.npy")
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        pose = torch.tensor(self.data[ind, :34]).float()
        action = torch.tensor(self.data[ind, 34]).long()
        return pose, action

if __name__=="__main__":
    """
    Testing code meant to check that the data loader works
    as intended.
    """

    train_data = PoseDataset("datasets/jackrabbot", actions=ACTIONS)
    test_data = PoseDataset("datasets/jackrabbot", actions=ACTIONS, partition="val")

    train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        num_workers = 0,
        batch_size = 4,
        pin_memory = True,
        shuffle = True
    )

    for i, data in enumerate(train_loader):
        pose, action = data
        print(pose.shape, action.shape)
        break
