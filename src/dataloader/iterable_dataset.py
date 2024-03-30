import torch
import math


def map_to_class(data, task='classification'):
    # x = (reference, alternate, annotation)
    # map x annotation to corresponding class label
    # y = target， e.g. beneign or pathogenic, slope, p_val, splice_change
    # for classification task, map y to corresponding class label
    # for regression task, keep y as it is
    # save the mapping as a file
    x_class = {}
    y_class = {}
    x_count = 0
    y_count = 0
    for i in range(len(data)):
        element = data[i]
        x, y = element
        annotation = x[2]
        if annotation not in x_class:
            x_class[annotation] = x_count
            x_count += 1
        x[2] = x_class[annotation]
        if task == 'classification':
            if y not in y_class:
                y_class[y] = y_count
                y_count += 1
            element[1] = y_class[y]
    return x_class, y_class


def transform(data):
    # transfrom y to torch.tensor
    x, y = data
    y = torch.tensor(y)
    return x, y

class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, task='classification', transform=None):
        self.x_class_mapping, self.y_class_mapping = map_to_class(data, task=task)
        print(f"x_class_mapping: {self.x_class_mapping}")
        print(f"y_class_mapping: {self.y_class_mapping}")
        self.data = data
        self.transform = transform
        print(f"Total data: {len(self.data)}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.data)
            print(f"iter_start: {iter_start}, iter_end: {iter_end}")
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
