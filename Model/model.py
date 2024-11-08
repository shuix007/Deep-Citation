import os
import torch
import torch.nn as nn

from typing import List
from transformers import AutoModel

def mask_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LanguageModel(nn.Module):
    def __init__(self, 
                 modelname: str, 
                 device: str, 
                 readout: str
        ):
        super(LanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname
        self.readout_fn = readout

        self.model = AutoModel.from_pretrained(modelname)
        self.hidden_size = self.model.config.hidden_size

    def readout(self, model_inputs, model_outputs, readout_masks=None):
        if self.readout_fn == 'cls':
            if 'bert' in self.modelname:
                text_representations = model_outputs.last_hidden_state[:, 0]
            elif 'xlnet' in self.modelname:
                text_representations = model_outputs.last_hidden_state[:, -1]
            else:
                raise ValueError('Invalid model name {} for the cls readout.'.format(self.modelname))
        elif self.readout_fn == 'mean':
            text_representations = mask_pooling(model_outputs, model_inputs['attention_mask'])
        elif self.readout_fn == 'ch' and readout_masks is not None:
            text_representations = mask_pooling(model_outputs, readout_masks)
        else:
            raise ValueError('Invalid readout function.')
        return text_representations

    def _lm_forward(self, tokens):
        tokens = tokens.to(self.device)
        if 'readout_mask' in tokens:
            readout_mask = tokens.pop('readout_mask')
        else:
            readout_mask = None
        outputs = self.model(**tokens)
        return self.readout(tokens, outputs, readout_mask)

    def forward(self):
        raise NotImplementedError

    def save_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        torch.save(self.state_dict(), model_filename)

    def load_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        self.load_state_dict(torch.load(model_filename))

class MultiHeadLanguageModel(LanguageModel):
    def __init__(self, 
                 modelname: str, 
                 device: str, 
                 readout: str, 
                 num_classes: List
        ):
        super().__init__(
            modelname,
            device, 
            readout
        )

        self.num_classes = num_classes
        self.lns = nn.ModuleList([nn.Linear(self.hidden_size, num_class) for num_class in num_classes])

    def forward(self, input_tokens, input_head_indices, class_tokens, class_head_indices):
        head_indices = torch.unique(input_head_indices)
        text_representations = self._lm_forward(input_tokens)

        final_preds = {}
        for i in head_indices:
            if torch.any(input_head_indices == i):
                final_preds[i.item()] = self.lns[i.item()](text_representations[input_head_indices == i])
            else:
                final_preds[i.item()] = torch.tensor([]).to(self.device)
        return final_preds