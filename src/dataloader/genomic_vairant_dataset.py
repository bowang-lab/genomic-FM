import torch
import math


# define a transform function
def transform(data):
    return data

class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.data)
        else:
            per_worker = int(math.ceil(len(self.data) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data))
        return iter(self.data[iter_start:iter_end])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data

    def __call__(self, idx):
        return self.__getitem__(idx)
