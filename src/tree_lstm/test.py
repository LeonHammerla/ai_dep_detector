from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

# IMPORT CONSTANTS
from sklearn.metrics import classification_report

from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm.model import SimilarityTreeLSTM
# DATA HANDLING CLASSES
from treelstm.vocab import Vocab
# DATASET CLASS FOR DATASET
from treelstm.dataset import TreeDataset
# METRICS CLASS FOR EVALUATION
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm.trainer import Trainer
# CONFIG PARSER
from config import parse_args


# MAIN BLOCK
def main():
    global args
    args = parse_args()
    # global logger

    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        # torch.backends.
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    test_dir = os.path.join(args.data, 'test/')

    # write unique words from all token files
    sick_vocab_file = os.path.join(args.data, 'sick.vocab')
    if not os.path.isfile(sick_vocab_file):
        token_files_a = [os.path.join(split, 'a.toks') for split in [test_dir]]
        token_files = token_files_a
        sick_vocab_file = os.path.join(args.data, 'sick.vocab')
        utils.build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])


    test_file = os.path.join(args.data, 'sick_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = TreeDataset(test_dir, vocab)
        torch.save(test_dataset, test_file)

    checkpoint = torch.load('%s.pt' % os.path.join(args.save, args.expname))
    # args = checkpoint["args"]
    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
        vocab.size(),
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.sparse,
        args.freeze_embed)
    model.load_state_dict(checkpoint['model'])
    # criterion = nn.KLDivLoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    model.emb.weight.data.copy_(emb)

    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    # optimizer.load_state_dict(checkpoint["optim"])
    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer, device)

    test_loss, test_acc, test_predictions, test_labels = trainer.test(test_dataset)
    print(classification_report(test_labels, test_predictions, target_names=["human", "machine"]))


if __name__ == "__main__":
    main()
