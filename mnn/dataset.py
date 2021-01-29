import numpy as np
import pickle
import MNN
from utils import utils
F = MNN.expr

word_dict = pickle.load(open('../vocab/word_index.pickle', 'rb'))
rev_word_dict = pickle.load(open('../vocab/index_word.pickle', 'rb'))
OOV_INDEX = len(word_dict) + 1
def load_data(filename, max_len):
    training_data = []
    with open(filename, "r") as fr:
        for line in fr:
            line = line.strip().split("\t")[0]
            sequence = [word_dict[w] if w in word_dict.keys() else OOV_INDEX for w in utils.get_word_list(line)]
            n = min(len(sequence), max_len)
            tmp = []
            for i in range(0, n - 1):
                tmp.append(sequence[i])
                a = []
                a.extend(tmp)
                training_data.append((utils.padding(a, max_len), sequence[i + 1]))

    return training_data

class Dataset(MNN.data.Dataset):
    def __init__(self, filename, max_len=8, is_training=True):
        super(Dataset, self).__init__()
        self.is_training = is_training
        self.filename = filename
        self.max_len = max_len
        if self.is_training:
            self.data_list = load_data(self.filename, max_len=self.max_len)

    def __getitem__(self, index):
        x = self.data_list[index][0]
        y = self.data_list[index][1]
        dl = F.const([y], [], F.data_format.NHWC, F.dtype.int)
        dv = F.const([x], [self.max_len], F.data_format.NHWC, F.dtype.int)


        return [dv], [dl]

    def __len__(self):
        if self.is_training:
            return len(self.data_list)
        else:
            return 0

