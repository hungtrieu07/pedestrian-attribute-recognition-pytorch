import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pickle
import copy

class AttDataset(data.Dataset):
    """
    person attribute dataset interface
    """
    def __init__(
        self, 
        dataset,
        partition,
        split='train',
        partition_idx=0,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        if os.path.exists(dataset):
            # Use 'rb' mode for Python 3 pickle loading
            with open(dataset, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            print(dataset + ' does not exist in dataset.')
            raise ValueError

        if os.path.exists(partition):
            # Use 'rb' mode for Python 3 pickle loading
            with open(partition, 'rb') as f:
                self.partition = pickle.load(f)
        else:
            print(partition + ' does not exist in dataset.')
            raise ValueError

        if split not in self.partition:
            print(split + ' does not exist in dataset.')
            raise ValueError

        if partition_idx > len(self.partition[split]) - 1:
            print('partition_idx is out of range in partition.')
            raise ValueError

        self.transform = transform
        self.target_transform = target_transform

        # create image & label based on the selected partition and dataset split
        self.root_path = self.dataset['root']
        self.att_name = [
            self.dataset['att_name'][i] for i in self.dataset['selected_attribute']
        ]
        self.image = []
        self.label = []

        for idx in self.partition[split][partition_idx]:
            self.image.append(self.dataset['image'][idx])
            label_tmp = np.array(self.dataset['att'][idx])[self.dataset['selected_attribute']].tolist()
            self.label.append(label_tmp)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the index of the target class
        """
        imgname, target = self.image[index], self.label[index]
        
        # load image and labels
        img_path = os.path.join(self.dataset['root'], imgname)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        # default no transform on target
        target = np.array(target).astype(np.float32)
        target[target == 0] = -1
        target[target == 2] = 0
        if self.target_transform is not None:
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.image)
