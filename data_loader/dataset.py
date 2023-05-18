from torch.utils.data import Dataset

'''
CustomDataset 클래스를 정의할 때, __init__, __len__, __getitem__ 함수를 정의해야 한다.
'''
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)

        return x, y
