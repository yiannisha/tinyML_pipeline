import os
import torch

from utils.iegm import txt_to_numpy, loadCSV

from _types import Mode, BaseDataset

class Dataset(BaseDataset):
    def __init__(self, root_dir: str, mode: Mode):
        self.name = 'iegm'
        self.root_dir = os.path.join(root_dir, self.name)
        self.names_list = []
        self.transform = None
        self.size = 1250

        csvdata_all = loadCSV(os.path.join(self.root_dir, f'{mode}_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = os.path.join(self.root_dir, 'data_set', self.names_list[idx].split(' ')[0])

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'data': torch.from_numpy(IEGM_seg), 'label': label}

        return sample

