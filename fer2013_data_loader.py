import os
import pickle

import numpy as np
import pandas as pd
import PIL.Image
import torch


class FER2013((torch.utils.data.Dataset)):
    """FER2013 Dataset.
        Args:
            _root, str: Root directory of dataset.
            _phase ['train'], str: train/val/test.
            _transform [None], function: A transform for a PIL.Image
            _target_transform [None], function: A transform for a label.
            _train_data, np.ndarray of shape N*3*48*48.
            _train_labels, np.ndarray of shape N.
            _val_data, np.ndarray of shape N*3*48*48.
            _val_labels, np.ndarray of shape N.
            _test_data, np.ndarray of shape N*3*48*48.
            _test_labels, np.ndarray of shape N.
        """

    def __init__(self, root, phase='train', transform=None,
                 target_transform=None):
        self._root = os.path.expanduser(root)
        self._phase = phase
        self._transform = transform
        self._target_transform = target_transform

        if (os.path.isfile(os.path.join(root, 'processed', 'train.pkl'))
                and os.path.isfile(os.path.join(root, 'processed', 'val.pkl'))
                and os.path.isfile(os.path.join(root, 'processed', 'test.pkl'))):
            print('Dataset already processed.')
        else:
            self.process('train', 28709)
            self.process('val', 3589)
            self.process('test', 3589)

        if self._phase == 'train':
            self._train_data, self._train_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'train.pkl'), 'rb'))
        elif self._phase == 'val':
            self._val_data, self._val_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'val.pkl'), 'rb'))
        elif self._phase == 'test':
            self._test_data, self._test_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'test.pkl'), 'rb'))
        else:
            raise ValueError('phase should be train/val/test.')