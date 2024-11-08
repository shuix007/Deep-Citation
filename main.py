import os
# change the default cache dir so that huggingface won't take the cse space.
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme1n1/zeren/HuggingfaceCache/'

from utils import save_args
from trainer import MultiHeadTrainer
from data import create_data_channels, create_single_data_object, MultiHeadDatasets
from Model import MultiHeadLanguageModel
import numpy as np
import random
import torch
import argparse

N_CLASSES = {
    'kim': 3,
    'acl': 6,
    'scicite': 3
}

def main_mtl(args):
    datasets = args.dataset.split('-')
    lambdas = [float(l) for l in args.lambdas.split('-')]

    if lambdas[0] != 1:
        lambdas[0] = 1.
        print('The first lambda is set to 1.')
    assert len(datasets) == len(lambdas), "The size of lambdas should be the same as the number of datasets."

    data_filenames = [os.path.join(args.data_dir, ds+'.tsv') for ds in datasets]

    if args.lm == 'scibert':
        modelname = 'allenai/scibert_scivocab_uncased'
    elif args.lm == 'bert':
        modelname = 'bert-base-uncased'
    else:
        modelname = args.lm

    train_data, val_data, test_data, model_label_map = create_data_channels(
        data_filenames[0],
        args.class_definition,
        lmbd=lambdas[0]
    )
    train_datasets_list = [train_data]
    if len(data_filenames) > 1:
        for i, data_filename in enumerate(data_filenames[1:]):
            aux_data, aux_label_map = create_single_data_object(
                data_filename, args.class_definition, split='train', lmbd=lambdas[i+1]
            )
            train_datasets_list.append(aux_data)
    train_datasets = MultiHeadDatasets(train_datasets_list, batch_size_factor=args.batch_size_factor)
    if train_datasets.adjusted_batch_size_factor > 1:
        args.batch_size = int(args.batch_size * train_datasets.adjusted_batch_size_factor)
        print('Adjusting the training batch size to {}.'.format(args.batch_size))

    model = MultiHeadLanguageModel(
        modelname=modelname,
        device=args.device,
        readout=args.readout,
        num_classes=[N_CLASSES[ds] for ds in datasets]
    ).to(args.device)

    finetuner = MultiHeadTrainer(
        model,
        train_datasets,
        val_data,
        test_data,
        args
    )
    
    if not args.inference_only:
        print('Finetuning.')
        finetuner.train()

    finetuner.load_model()
    print('Evaluating best val checkpoint.')
    preds = finetuner.test()
    print('Lambdas: {}'.format(lambdas))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--lambdas', required=True)
    parser.add_argument('--data_dir', default='Data/', type=str)
    parser.add_argument('--workspace', default='Workspaces/Test', type=str)
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
    parser.add_argument('--seed', default=1, type=int)

    # model configuration
    parser.add_argument('--lm', default='scibert', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--batch_size_factor', default=2, type=int)
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
    main_mtl(args)