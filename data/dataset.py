import torch
from torch.utils.data import Dataset, DataLoader
import json, os
import cfg
import pdb


class SummarizationDataset(Dataset):
    '''
    Dataset for loading articles and target summary
    '''
    def __init__(self, txt_path, emb_path, map_path):
        """
        txt_path: path to txt file
        emb_path: path to word2vec embeddings
        map_path: path to vocab to embeddings mapping file
        """

        if not os.path.isfile(txt_path):
            raise Exception("Data file not found at location \"{}\".".format(txt_path))
        if not os.path.isfile(emb_path):
            raise Exception("Embedding file not found at location \"{}\".".format(emb_path))
        if not os.path.isfile(map_path):
            raise Exception("JSON mapping file not found at location \"{}\".".format(map_path))

        print("Loading txt file into memory...")
        self.data = []
        self.target = []

        with open(txt_path, 'r') as reader:
            for i, line in enumerate(reader):
                tokens = line.split()

                if i % 2 == 0:
                    self.data.append(tokens)
                elif i % 2 == 1:
                    self.target.append(tokens)
        print("Finished loading txt file.")

        if not len(self.data) == len(self.target):
            raise Exception("Number of articles do not match the number of summaries.")

        if not len(self.data) == len(self.target):
            raise Exception("JSON mapping file not found at location \"{}\".".format(map_path))

        self.emb = torch.load(emb_path)

        if not self.emb.size() == torch.Size([cfg.VOCAB_SIZE + 2, cfg.EMBEDDING_SIZE]):
            raise Exception("Embeddings do not have the correct shape.")

        with open(map_path, 'r') as json_file:
            self.map = json.load(json_file)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


        data = self.data[idx]
        target = self.target[idx]

        input_emb = torch.zeros((len(data), cfg.EMBEDDING_SIZE))

        word_indices = []
        for token in data:
            if token in self.map.keys():
                word_indices.append(self.map[token])
            else:
                word_indices.append(-1)

        for i, word_idx in enumerate(word_indices):
            if word_idx != -1:
                input_emb[i] = self.emb[word_idx]

        target_idx = torch.zeros((len(target), cfg.VOCAB_SIZE + 2))

        for i, token in enumerate(target):
            target_idx[i, self.map[token]] = 1

        return input_emb, target_idx


def output2tokens(index, idx2word):
    """
    index: a (N x V) Tensor where N is the length of the sentence and V the length of vocab + 2
    idx2word: dict that maps embedding index to tokens
    """
    tokens = []

    for i in range(index.size()[0]):
        _, idx = index[i].max(0)
        tokens.append(idx2word[str(idx.item())])

    return tokens

def get_dataloader(dataset):
    dataloader = DataLoader(dataset, shuffle=True, num_workers=1)
    return dataloader

# For testing
if __name__ == "__main__":

    with open('idx2word.json', 'r') as json_file:
        idx2word = json.load(json_file)

    
    dataset = SummarizationDataset(os.path.join('finished', 'val.txt'),
                                   'GloVe_embeddings.pt',
                                   'word2idx.json')

    for i in range(len(dataset)):
        input_emb, target_idx = dataset[i]
        tokens = output2tokens(target_idx, idx2word)
        print(tokens)
        pdb.set_trace()