import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from scipy.special import softmax
from sklearn.metrics import f1_score, confusion_matrix
from data import CollateFn
from scheduler import get_slanted_triangular_scheduler, get_linear_scheduler


class Trainer(object):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.tol = args.tol

        self.workspace = args.workspace
        self.model = model
        self.modelname = args.lm

        self.load(train_dataset, val_dataset, test_dataset)

    def load(self, train_dataset, val_dataset, test_dataset):
        self.load_datasets(train_dataset, val_dataset, test_dataset)
        self.load_losses()
        self.load_optimizer()
        self.load_scheduler()
        self.load_logger()

    def load_losses(self):
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def load_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), 
                               lr=self.args.lr, weight_decay=self.args.l2)
    
    def load_datasets(self, train_dataset, val_dataset, test_dataset):
        # modelname = 'allenai/scibert_scivocab_uncased' if self.args.lm == 'scibert' else 'bert-base-uncased'
        if self.args.lm == 'scibert':
            modelname = 'allenai/scibert_scivocab_uncased'
        elif self.args.lm == 'bert':
            modelname = 'bert-base-uncased'
        else:
            modelname = self.args.lm
        self.collate_fn = CollateFn(modelname=modelname, instance_weights=True)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def load_logger(self):
        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))
        self.best_epoch = 0
        self.best_metric = 0.

    def load_scheduler(self):
        if self.args.scheduler == 'exp':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.decay_step, gamma=self.args.decay_rate, verbose=True)
        elif self.args.scheduler == 'slanted':
            self.scheduler = get_slanted_triangular_scheduler(
                self.optimizer, num_epochs=self.num_epochs)
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_scheduler(
                self.optimizer, num_training_epochs=self.num_epochs, initial_lr=self.args.lr, final_lr=self.args.lr/32)
        elif self.args.scheduler == 'const':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=self.num_epochs, gamma=1., verbose=True)
        else:
            raise ValueError(
                'Scheduler {} not implemented.'.format(self.args.scheduler))
        print('Using {} scheduler.'.format(self.args.scheduler))

    def update_train_dataset(self, train_dataset):
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def compute_loss(self, labels, logits, ds_indices, instance_weights):
        losses = []
        for ds_idx in logits.keys():
            if len(logits[ds_idx]) > 0:
                losses.append(
                    (self.loss_fn(logits[ds_idx], labels[ds_indices == ds_idx]) * instance_weights[ds_indices == ds_idx]).sum() / instance_weights[ds_indices == ds_idx].sum()
                )
        loss = sum(losses)
        return loss

    def compute_metrics(self, labels, logits):
        return f1_score(labels, logits.argmax(axis=1), average='macro')

    def early_stop(self, metric, epoch):
        cur_metric = sum(metric) if type(metric) == list else metric

        if cur_metric > self.best_metric:
            self.best_metric = cur_metric
            self.best_epoch = epoch
            self.save_model()

        return (epoch - self.best_epoch >= self.tol)

    def train_one_epoch(self):
        total_loss = 0.
        self.model.train()

        for batched_text, labels, ds_indices, instance_weights in self.train_dataloader:
            labels = labels.to(self.device)
            ds_indices = ds_indices.to(self.device)
            instance_weights = instance_weights.to(self.device)

            preds = self.model(batched_text, ds_indices)
            loss = self.compute_loss(labels, preds, ds_indices, instance_weights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def eval_one_epoch(self, return_preds=False, test=False, outside_dataloader=None):
        self.model.eval()

        if outside_dataloader is not None:
            data_loader = outside_dataloader
        else:
            if test:
                data_loader = self.test_dataloader
            else:
                data_loader = self.val_dataloader  # for validation

        preds, labels = list(), list()
        with torch.no_grad():
            for batched_text, label, ds_indices, instance_weights in data_loader:
                preds.append(self.model(
                    batched_text, ds_indices
                )[0])
                labels.append(label)
            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).numpy()
            metric = self.compute_metrics(labels, preds)

            if return_preds:
                return metric, preds
            return metric

    def train(self):
        for epoch in range(1, self.num_epochs+1):
            start = time.time()
            loss = self.train_one_epoch()
            end_train = time.time()
            metric = self.eval_one_epoch(test=False)
            end = time.time()

            self.log_tensorboard(metric, loss, epoch)
            self.log_print(metric, loss, epoch, end_train-start, end-end_train)
            self.scheduler.step()

            if self.early_stop(metric=metric, epoch=epoch):
                break

    def test(self, outside_dataset=None):
        if outside_dataset is not None:
            outside_dataloader = DataLoader(
                outside_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
            _, test_preds = self.eval_one_epoch(
                return_preds=True, outside_dataloader=outside_dataloader, test=False)
        else:
            test_metric, test_preds = self.eval_one_epoch(return_preds=True, test=True)
            print('Test results:')
            print('Test metric: {:.4f}.'.format(test_metric))
            print('Best val metric: {:.4f}'.format(self.best_metric))
        return test_preds

    def save_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        torch.save(self.model.state_dict(), model_filename)

    def load_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        self.model.load_state_dict(torch.load(model_filename))

    def checkpoint(self, epoch):
        checkpoint_filename = os.path.join(self.workspace, 'checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, checkpoint_filename)

    def load_checkpoint(self):
        checkpoint_filename = os.path.join(self.workspace, 'checkpoint.pt')
        state_dict = torch.load(checkpoint_filename)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        return state_dict['epoch']

    def log_tensorboard(self, metric, loss, epoch):
        ''' Write experiment log to tensorboad. '''
        cur_metric = metric[0] if type(metric) == list else metric

        self.logger.add_scalar('Training loss', loss, epoch)
        self.logger.add_scalar('ROC', cur_metric, epoch)

    def log_print(self, metric, loss, epoch, train_time, eval_time):
        ''' Stdout of experiment log. '''

        if type(metric) == list:
            main_metric = metric[0]
            aux_metrics = ', '.join(['%.4f' % r for r in metric[1:]])
            print('Epoch: {}, train time: {:.4f}, eval time: {:.4f}, training loss: {:.4f}, main val metric: {:.4f}, aux metric: {}'.format(
                epoch, train_time, eval_time, loss, main_metric, aux_metrics))
        elif type(metric) in [float, np.float64]:
            print('Epoch: {}, train time: {:.4f}, eval time: {:.4f}, training loss: {:.4f}, val metric: {:.4f}'.format(
                epoch, train_time, eval_time, loss, metric))
        else:
            raise TypeError('Type metric {} is wrong.'.format(type(metric)))

class MultiHeadTrainer(Trainer):
    def load_datasets(self, train_datasets, val_dataset, test_dataset):
        class_definitions = train_datasets.class_definitions
        modelname = 'allenai/scibert_scivocab_uncased' if self.args.lm == 'scibert' else 'bert-base-uncased'
        self.collate_fn = CollateFn(modelname=modelname, class_definitions=class_definitions, instance_weights=False)

        self.lambdas = train_datasets.lambdas
        self.lambdas[0] = 1.

        self.train_dataloader = DataLoader(
            train_datasets, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=self.collate_fn)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        
    def load_losses(self):
        num_losses = len(torch.unique(self.collate_fn.class_head_indices))
        self.loss_fns = [nn.CrossEntropyLoss() for _ in range(num_losses)]

    def compute_loss(self, labels, logits, ds_indices):
        losses = []
        for ds_idx in logits.keys():
            if len(logits[ds_idx]) > 0:
                losses.append(
                    self.lambdas[ds_idx] * self.loss_fns[ds_idx](logits[ds_idx], labels[ds_indices == ds_idx])
                )
        loss = sum(losses)
        return loss
    
    def train_one_epoch(self):
        total_loss = 0.
        self.model.train()

        for batched_text, labels, ds_indices, class_tokens, class_ds_indices in self.train_dataloader:
            labels = labels.to(self.device)
            ds_indices = ds_indices.to(self.device)
            class_ds_indices = class_ds_indices.to(self.device)

            preds = self.model(batched_text, ds_indices, class_tokens, class_ds_indices)
            loss = self.compute_loss(labels, preds, ds_indices)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def eval_one_epoch(self, return_preds=False, test=False, outside_dataloader=None):
        self.model.eval()

        if outside_dataloader is not None:
            data_loader = outside_dataloader
        else:
            if test:
                data_loader = self.test_dataloader
            else:
                data_loader = self.val_dataloader  # for validation

        preds, labels = list(), list()
        with torch.no_grad():
            for batched_text, label, ds_indices, class_tokens, class_ds_indices in data_loader:
                preds.append(self.model(
                    batched_text, ds_indices, class_tokens, class_ds_indices
                )[0])
                labels.append(label)
            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).numpy()
            metric = self.compute_metrics(labels, preds)

            if return_preds:
                return metric, preds
            return metric