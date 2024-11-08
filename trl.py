import os

from utils import save_args
from trainer import MultiHeadTrainer
from data import create_data_channels, create_single_data_object, MultiHeadDatasets
from Model import MultiHeadLanguageModel
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
import random
import torch
import argparse

N_CLASSES = {
    'kim': 3,
    'acl': 6,
    'scicite': 3
}

def trl(labels, pred_logits):
    N = len(labels)
    preds = pred_logits.argmax(axis=1)

    label_base = len(np.unique(labels))
    pred_base = pred_logits.shape[1]
    confusion_matrix = np.zeros((label_base, pred_base))
    for i in range(N):
        # confusion_matrix[labels[i], preds[i]] += 1
        confusion_matrix[labels[i]] += softmax(pred_logits[i])
    # print(confusion_matrix)

    base_entropy = entropy(confusion_matrix.sum(axis=1) / confusion_matrix.sum(), base=label_base)
    pred_entropy = (entropy(confusion_matrix / confusion_matrix.sum(axis=0, keepdims=True), axis=0, base=label_base) * ((confusion_matrix.sum(axis=0) / confusion_matrix.sum()))).sum()
    lambda_ = (base_entropy - pred_entropy) / base_entropy
    print('Base entropy: {:.4f}, pred entropy: {:.4f}, lambda: {:.4f}'.format(base_entropy, pred_entropy, lambda_))

def main_trl(args):
    primary_dataset = args.primary_dataset
    auxiliary_dataset = args.auxiliary_dataset

    primary_data_filename = os.path.join(args.data_dir, primary_dataset+'.tsv')
    auxiliary_data_filename = os.path.join(args.data_dir, auxiliary_dataset+'.tsv')

    if args.lm == 'scibert':
        modelname = 'allenai/scibert_scivocab_uncased'
    elif args.lm == 'bert':
        modelname = 'bert-base-uncased'
    else:
        modelname = args.lm

    model = MultiHeadLanguageModel(
        modelname=modelname,
        device=args.device,
        readout=args.readout,
        num_classes=[N_CLASSES[auxiliary_dataset]]
    ).to(args.device)

    train_data, val_data, test_data, model_label_map = create_data_channels(
        auxiliary_data_filename,
        args.class_definition,
        lmbd=1.
    )
    train_datasets_list = [train_data]
    train_datasets = MultiHeadDatasets(train_datasets_list)

    finetuner = MultiHeadTrainer(
        model,
        train_datasets,
        val_data,
        test_data,
        args
    )
    finetuner.train()
    finetuner.load_model()
    preds = finetuner.test()

    aux_data, aux_label_map = create_single_data_object(
        primary_data_filename, args.class_definition, split='train', lmbd=1.
    )
    aux_preds = finetuner.test(outside_dataset=aux_data)
    aux_labels = aux_data.original_labels.numpy()
    print("Primary: {}, Auxiliary: {}".format(primary_dataset, auxiliary_dataset))
    trl(aux_labels, aux_preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--primary_dataset', default='acl', type=str)
    parser.add_argument('--auxiliary_dataset', default='scicite', type=str)
    parser.add_argument('--data_dir', default='Data/', type=str)
    parser.add_argument('--workspace', default='Workspaces', type=str)
    parser.add_argument('--class_definition', default='Data/class_def.json', type=str)

    # training configuration
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--decay_rate', default=0.5, type=float)
    parser.add_argument('--decay_step', default=5, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--scheduler', default='slanted', type=str)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--tol', default=10, type=int)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--use_abstract', action='store_true')
    parser.add_argument('--seed', default=42, type=int)  # seed = 1209384756

    # model configuration
    parser.add_argument('--lm', default='scibert', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--readout', default='ch', type=str)

    args = parser.parse_args()

    # fix all random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True

    # save the arguments
    if not os.path.exists(args.workspace):
        os.mkdir(args.workspace)
    save_args(args, args.workspace)

    main_trl(args)