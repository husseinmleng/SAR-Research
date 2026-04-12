import os
import time
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision as tv
from torchvision import transforms, datasets


class self_Dataset(Dataset):
    def __init__(self, data, label=None):
        super(self_Dataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        if self.label is not None:
            return data, self.label[index]
        return data, 1

    def __len__(self):
        return len(self.data)


def count_data(data_dict):
    return sum(len(v) for v in data_dict.values())


class self_DataLoader(Dataset):
    def __init__(self, root, train=True, dataset='MSTAR', seed=1, nway=5,
                 unseen_class='T72', unseen_ratio=1.0,
                 gan_augment=False, gan_output_dir='gan_output',
                 augment_rotation=False, augment_speckle=False, speckle_sigma=0.1):
        super(self_DataLoader, self).__init__()

        self.seed = seed
        self.nway = nway
        self.unseen_class = unseen_class
        # probability of picking an unseen query: ratio / (1 + ratio)
        self.unseen_prob = unseen_ratio / (1.0 + unseen_ratio)

        self.gan_augment = gan_augment
        self.gan_output_dir = gan_output_dir

        self.augment_rotation = augment_rotation
        self.augment_speckle = augment_speckle
        self.speckle_sigma = speckle_sigma

        self.transform_SAR = tv.transforms.Compose([
            tv.transforms.Grayscale(1),
            tv.transforms.Resize((100, 100)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.full_train_dict, self.full_test_dict, self.few_data_dict = \
            self.load_data(root, dataset)

        # Optionally load GAN-generated images for support augmentation
        self.gan_data_dict = {}
        if self.gan_augment:
            self._load_gan_data()

        print('full_train_num: %d' % count_data(self.full_train_dict))
        print('full_test_num:  %d' % count_data(self.full_test_dict))
        print('few_data_num:   %d' % count_data(self.few_data_dict))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, root, dataset):
        if dataset != 'MSTAR':
            raise NotImplementedError('Only MSTAR dataset is supported')

        # Classes are directly under root (not in a MSTAR/ sub-folder)
        train_dataset = datasets.ImageFolder(
            root=root,
            transform=self.transform_SAR
        )

        # Resolve unseen class index by name
        if self.unseen_class not in train_dataset.class_to_idx:
            available = sorted(train_dataset.class_to_idx.keys())
            raise KeyError(
                f'unseen_class "{self.unseen_class}" not found in dataset. '
                f'Available classes: {available}'
            )
        few_selected_label = [train_dataset.class_to_idx[self.unseen_class]]
        print('unseen class: %s (idx %d)' % (self.unseen_class, few_selected_label[0]))

        full_data_dict = {}
        few_data_dict = {}

        loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
        for data, label in loader:
            label = label.item()
            data = data.squeeze(0)
            target = few_data_dict if label in few_selected_label else full_data_dict
            target.setdefault(label, []).append(data)

        # 80 / 20 train / test split on seen classes
        keys = list(full_data_dict.keys())
        train_lists = [list(full_data_dict[k]) for k in keys]
        test_lists  = [list(full_data_dict[k]) for k in keys]

        for i in range(len(keys)):
            n = int(len(train_lists[i]) * 0.8)
            test_lists[i]  = train_lists[i][n:]
            train_lists[i] = train_lists[i][:n]

        full_train_dict = dict(zip(keys, train_lists))
        full_test_dict  = dict(zip(keys, test_lists))

        return full_train_dict, full_test_dict, few_data_dict

    def _load_gan_data(self):
        """Load GAN-generated images from gan_output_dir/<class_name>/."""
        gan_transform = tv.transforms.Compose([
            tv.transforms.Grayscale(1),
            tv.transforms.Resize((100, 100)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))
        ])
        if not os.path.isdir(self.gan_output_dir):
            print(f'[WARNING] gan_output_dir "{self.gan_output_dir}" not found, '
                  'GAN augmentation disabled.')
            self.gan_augment = False
            return

        gan_dataset = datasets.ImageFolder(
            root=self.gan_output_dir,
            transform=gan_transform
        )
        loader = torch.utils.data.DataLoader(gan_dataset, shuffle=True)
        for data, label in loader:
            label = label.item()
            data = data.squeeze(0)
            self.gan_data_dict.setdefault(label, []).append(data)

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _maybe_augment(self, tensor):
        from augment import RandomRotation360, SpeckleNoise
        if self.augment_rotation:
            tensor = RandomRotation360()(tensor)
        if self.augment_speckle:
            tensor = SpeckleNoise(self.speckle_sigma)(tensor)
        return tensor

    def _augment_list(self, tensors):
        return [self._maybe_augment(t) for t in tensors]

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def load_batch_data(self, train=True, batch_size=16, nway=5, num_shots=1):
        if train:
            return self._load_train_batch(batch_size, nway, num_shots)
        return self._load_test_batch(batch_size, nway, num_shots)

    def _load_train_batch(self, batch_size, nway, num_shots):
        data_dict = self.full_train_dict

        x, label_y, one_hot_y, class_y = [], [], [], []
        xi, label_yi, one_hot_yi, map_label2class = [], [], [], []

        for i in range(batch_size):
            sampled_classes = random.sample(list(data_dict.keys()), nway + 1)

            # Seen or unseen query based on unseen_prob
            if random.random() < self.unseen_prob:
                positive_class = nway   # unseen slot
            else:
                positive_class = random.randint(0, nway - 1)

            label2class = torch.LongTensor(nway + 1)
            single_xi, single_label_yi, single_one_hot_yi = [], [], []

            for j, _class in enumerate(sampled_classes):
                one_hot = torch.zeros(nway + 1)
                one_hot[j] = 1.0

                if positive_class == nway:
                    if j == positive_class:
                        sampled_data = random.sample(data_dict[_class], 1)
                        x.append(sampled_data[0])
                        label_y.append(torch.LongTensor([j]))
                        one_hot_y.append(one_hot)
                        class_y.append(torch.LongTensor([_class]))
                        shots_data = []
                    else:
                        shots_data = random.sample(data_dict[_class], num_shots)
                else:
                    if j == positive_class:
                        sampled_data = random.sample(data_dict[_class], num_shots + 1)
                        x.append(sampled_data[0])
                        label_y.append(torch.LongTensor([j]))
                        one_hot_y.append(one_hot)
                        class_y.append(torch.LongTensor([_class]))
                        shots_data = sampled_data[1:]
                    elif j != nway:
                        shots_data = random.sample(data_dict[_class], num_shots)
                    else:
                        continue

                # GAN augmentation: append synthetic shots up to num_shots
                if self.gan_augment and _class in self.gan_data_dict and len(shots_data) < num_shots:
                    extra = num_shots - len(shots_data)
                    gan_shots = random.sample(
                        self.gan_data_dict[_class],
                        min(extra, len(self.gan_data_dict[_class]))
                    )
                    shots_data = list(shots_data) + gan_shots

                if shots_data:
                    shots_data = self._augment_list(shots_data)
                    single_xi += shots_data
                    single_label_yi.append(torch.LongTensor([j]).repeat(len(shots_data)))
                    single_one_hot_yi.append(one_hot.unsqueeze(0).repeat(len(shots_data), 1))

                label2class[j] = _class

            if not single_xi:
                continue

            shuffle_index = torch.randperm(len(single_xi))
            xi.append(torch.stack(single_xi, dim=0)[shuffle_index])
            label_yi.append(torch.cat(single_label_yi, dim=0)[shuffle_index])
            one_hot_yi.append(torch.cat(single_one_hot_yi, dim=0)[shuffle_index])
            map_label2class.append(label2class)

        # Trim to batch_size in case of skips
        n = min(len(x), len(xi))
        x = x[:n]; label_y = label_y[:n]
        one_hot_y = one_hot_y[:n]; class_y = class_y[:n]
        xi = xi[:n]; label_yi = label_yi[:n]
        one_hot_yi = one_hot_yi[:n]; map_label2class = map_label2class[:n]

        return [
            torch.stack(x, 0),
            torch.cat(label_y, 0),
            torch.stack(one_hot_y, 0),
            torch.cat(class_y, 0),
            torch.stack(xi, 0),
            torch.stack(label_yi, 0),
            torch.stack(one_hot_yi, 0),
            torch.stack(map_label2class, 0)
        ]

    def _load_test_batch(self, batch_size, nway, num_shots):
        data_dict = self.full_test_dict
        test_data_dict = self.few_data_dict

        x, label_y, one_hot_y, class_y = [], [], [], []
        xi, label_yi, one_hot_yi, map_label2class = [], [], [], []

        for i in range(batch_size):
            sampled_classes = random.sample(list(data_dict.keys()), nway)
            test_class = random.choice(list(test_data_dict.keys()))

            if random.random() < 0.5:
                positive_class = nway
            else:
                positive_class = random.randint(0, nway - 1)

            label2class = torch.LongTensor(nway + 1)
            single_xi, single_label_yi, single_one_hot_yi = [], [], []

            if positive_class == nway:
                for j, _class in enumerate(sampled_classes):
                    shots_data = random.sample(data_dict[_class], num_shots)
                    one_hot = torch.zeros(nway + 1)
                    one_hot[j] = 1.0
                    single_xi += shots_data
                    single_label_yi.append(torch.LongTensor([j]).repeat(num_shots))
                    single_one_hot_yi.append(one_hot.repeat(num_shots, 1))
                    label2class[j] = _class

                sampled_data = random.sample(test_data_dict[test_class], 1)
                x.append(sampled_data[0])
                label_y.append(torch.LongTensor([nway]))
                one_hot = torch.zeros(nway + 1)
                one_hot[nway] = 1.0
                one_hot_y.append(one_hot)
                class_y.append(torch.LongTensor([10]))
                label2class[nway] = test_class

            else:
                for j, _class in enumerate(sampled_classes):
                    one_hot = torch.zeros(nway + 1)
                    one_hot[j] = 1.0
                    if j == positive_class:
                        sampled_data = random.sample(data_dict[_class], num_shots + 1)
                        x.append(sampled_data[0])
                        label_y.append(torch.LongTensor([j]))
                        one_hot_y.append(one_hot)
                        class_y.append(torch.LongTensor([_class]))
                        shots_data = sampled_data[1:]
                    else:
                        shots_data = random.sample(data_dict[_class], num_shots)

                    single_xi += shots_data
                    single_label_yi.append(torch.LongTensor([j]).repeat(num_shots))
                    single_one_hot_yi.append(one_hot.repeat(num_shots, 1))
                    label2class[j] = _class

            shuffle_index = torch.randperm(num_shots * nway)
            xi.append(torch.stack(single_xi, dim=0)[shuffle_index])
            label_yi.append(torch.cat(single_label_yi, dim=0)[shuffle_index])
            one_hot_yi.append(torch.cat(single_one_hot_yi, dim=0)[shuffle_index])
            map_label2class.append(label2class)

        return [
            torch.stack(x, 0),
            torch.cat(label_y, 0),
            torch.stack(one_hot_y, 0),
            torch.cat(class_y, 0),
            torch.stack(xi, 0),
            torch.stack(label_yi, 0),
            torch.stack(one_hot_yi, 0),
            torch.stack(map_label2class, 0)
        ]

    def load_tr_batch(self, batch_size=16, nway=5, num_shots=1):
        return self.load_batch_data(True, batch_size, nway, num_shots)

    def load_te_batch(self, batch_size=16, nway=1, num_shots=1):
        return self.load_batch_data(False, batch_size, nway, num_shots)

    def get_data_list(self, data_dict):
        data_list, label_list = [], []
        for k, samples in data_dict.items():
            for s in samples:
                data_list.append(s)
                label_list.append(k)
        now_time = time.time()
        random.Random(now_time).shuffle(data_list)
        random.Random(now_time).shuffle(label_list)
        return data_list, label_list
