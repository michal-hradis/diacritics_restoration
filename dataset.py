import os
import unicodedata
import numpy as np
from torch.utils.data import Dataset


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if not unicodedata.combining(c))


def keep_accents(s):
    return ''.join('o' if len(unicodedata.normalize('NFD', c)) == 1 else unicodedata.normalize('NFD', c)[1] for c in s)


def add_diacritics(base_string, diacritics):
    return ''.join(''.join([c1, c2]) if c2 != 'o' else c1 for c1, c2 in zip(base_string, diacritics))


def string_difference(s1, s2):
    return np.sum(c1 != c2 for c1, c2 in zip(s1, s2))


def get_charset(data):
    chars = set(''.join(data))
    return sorted(chars)


class Translation:
    def __init__(self, chars):
        self.chars = chars
        self.id_map = {c:i for i, c in enumerate(chars)}

    def to_numpy(self, s):
        return np.asarray([self.id_map[c] for c in s], dtype=np.uint8)

    def to_string(self, ar):
        return ''.join(self.chars[i] for i in ar)


class TextDataset(Dataset):
    def __init__(self, input_file, min_length=20, max_length=96, input_translator=None, target_translator=None):
        super().__init__()
        self.min_length = 20
        self.max_length = 96
        self.input_file = input_file

        self.data_orig = []
        self.data_input = []
        self.data_target = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                while len(line) > min_length:
                    if len(self.data_orig) % 1000 == 0:
                        print(f'Loaded {len(self.data_orig)} segments from {self.input_file}.')

                    segment = line[:max_length]
                    line = line[max_length:]
                    segment = segment + ''.join(' ' for i in range(self.max_length - len(segment)))
                    segment_input = strip_accents(segment)
                    if string_difference(segment_input, segment) < 2:
                        continue
                    segment_target = keep_accents(segment)
                    if len(segment) == len(segment_input) == len(segment_target):
                        self.data_orig.append(segment)
                        self.data_input.append(segment_input)
                        self.data_target.append(segment_target)
                    else:
                        print(f"Lengths don't match {len(segment)} {len(segment_input)} {len(segment_target)}")

        if input_translator is None:
            self.input_translator = Translation(get_charset(self.data_input))
            self.target_translator = Translation(get_charset(self.data_target))
        else:
            self.input_translator = input_translator
            self.target_translator = target_translator

        self.data_input = [self.input_translator.to_numpy(line) for line in self.data_input]
        self.data_target = [self.target_translator.to_numpy(line) for line in self.data_target]

    def get_name(self):
        name = os.path.basename(self.input_file)
        return name

    def __len__(self):
        return len(self.data_orig)

    def __getitem__(self, idx):
        return self.data_input[idx], self.data_target[idx]