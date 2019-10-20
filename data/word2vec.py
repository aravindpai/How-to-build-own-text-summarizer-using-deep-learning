from torchnlp.word_to_vector import GloVe
import os, sys
import unicodedata, re
import torch
import json
import cfg

# For debug
import pdb


VOCAB_SIZE = 200000

# Whether to re-train embeddings on current dataset
TRAIN = False


def main(vocab_file):

    GloVe_vectors = GloVe(name='6B', dim=cfg.EMBEDDING_SIZE)

    embeddings = torch.Tensor(cfg.VOCAB_SIZE + 2, 100)
    word2idx, idx2word = {}, {}
    word2idx[cfg.SENTENCE_START] = cfg.VOCAB_SIZE
    word2idx[cfg.SENTENCE_END] = cfg.VOCAB_SIZE + 1

    with open(vocab_file, 'r') as reader:

        for i, line in enumerate(reader):

            token, count = line.split(' ')

            embeddings[i] = GloVe_vectors[token]
            word2idx[token] = i
            idx2word[i] = token

    # Start and end tokens
    embeddings[-2] = torch.cat((torch.zeros(cfg.EMBEDDING_SIZE // 2),
                                torch.ones(cfg.EMBEDDING_SIZE // 2)), 0)
    embeddings[-1] = torch.cat((torch.ones(cfg.EMBEDDING_SIZE // 2),
                                torch.zeros(cfg.EMBEDDING_SIZE // 2)), 0)

    torch.save(embeddings, 'GloVe_embeddings.pt')
    word2idx = json.dumps(word2idx)
    idx2word = json.dumps(idx2word)

    with open("word2idx.json", 'w') as json_writer:
        json_writer.write(word2idx)

    with open("idx2word.json", 'w') as json_writer:
        json_writer.write(idx2word)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python word2vec.py <vocab_file>")
        sys.exit()

    vocab_file = sys.argv[1]

    if not os.path.isfile(vocab_file):
        raise Exception("Cannot find vocal file {}.".format(vocab_file))

    main(vocab_file)