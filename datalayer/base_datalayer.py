# -*- coding:utf-8 -*-
from torch.utils.data.dataset import Dataset


class BaseDataLayer(Dataset):
    def getDataLoader(self):
        raise NotImplementedError
