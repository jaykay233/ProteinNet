import math
import numpy as np
import dataset
from dataset import get_dataset, split_with_shuffle, get_data_labels
from utils import Q8_accuracy
from radam import RAdam
from torch.nn import functional

LR = 0.0005
drop_out = 0.3
bz = 32
nn_epochs = 20

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import copy
##
## choose_data_set
##
from cb513_process import train_set, val_set, test_set

train_loader = DataLoader(train_set, batch_size=bz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=bz, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=bz, shuffle=False, num_workers=4)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    # scores: batch_size, n_head,seq_len,seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        ## auto regressive
        attn_shape = (1, 3000, 3000)
        subsequent_mask = np.triu(np.ones(attn_shape), k=100).astype('uint8') + np.tril(np.ones(attn_shape),
                                                                                        k=-100).astype('uint8')
        self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1).cuda()

    def forward(self, x):
        """
        :param x:
        :return:
        """
        nbatches, seq_len, d_model = x.shape
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=self.mask[:, :, :seq_len, :seq_len],
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = torch.nn.Sequential(nn.Linear(dim * 2, 1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input1, input2):
        concated = torch.cat([input1, input2], dim=2)
        gate = self.linear(concated)
        return self.softmax(gate * input1 + (1 - gate) * input2)


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    # seq_range_expand = torch.tensor(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.reshape(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


data_dim = 21


class Inception(nn.Module):
    def __init__(self, channel_last=True):
        super(Inception, self).__init__()
        self.channel_last = channel_last

        self.conv5 = torch.nn.Sequential(nn.Conv1d(in_channels=42, out_channels=512, kernel_size=5, padding=2),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2))
        self.conv1A1 = clones(
            torch.nn.Sequential(nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
                                nn.BatchNorm1d(256)), 4)
        self.conv3A2 = clones(
            torch.nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                nn.BatchNorm1d(256)), 4)
        self.conv3A3 = clones(
            torch.nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                nn.BatchNorm1d(256)), 2)
        self.conv3A4 = clones(
            torch.nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                nn.BatchNorm1d(256)), 1)

        self.conv1B1 = clones(
            torch.nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=1, padding=0),
                                nn.BatchNorm1d(128)), 4)
        self.conv3B2 = clones(
            torch.nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                nn.BatchNorm1d(128)), 4)
        self.conv3B3 = clones(
            torch.nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                nn.BatchNorm1d(128)), 2)
        self.conv3B4 = clones(
            torch.nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                nn.BatchNorm1d(128)), 1)

        self.conv1C1 = clones(torch.nn.Sequential(nn.Conv1d(in_channels=512, out_channels=64, kernel_size=1, padding=0),
                                                  nn.BatchNorm1d(64)), 4)
        self.conv3C2 = clones(torch.nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                                  nn.BatchNorm1d(64)), 4)
        self.conv3C3 = clones(torch.nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                                  nn.BatchNorm1d(64)), 2)
        self.conv3C4 = clones(torch.nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                                  nn.BatchNorm1d(64)), 1)

        self.conv11 = torch.nn.Sequential(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=11, padding=5),
                                          nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11, padding=5),
                                          nn.Dropout(0.25))

        self.dense = torch.nn.Sequential(nn.Linear(128, 42), nn.Dropout(0.25), nn.Linear(42, 42))

    def forward(self, input):
        output = input
        if self.channel_last:
            output = output.transpose(1, 2)
        for i in range(len(self.conv5)):
            output = self.conv5[i](output)
        outputs = [conv(output) for conv in self.conv1A1]
        outputs[1] = self.conv3A2[0](outputs[1])
        outputs[2] = self.conv3A3[1](outputs[2])
        outputs[2] = self.conv3A4[0](outputs[2])
        output = torch.cat(outputs, dim=1)

        outputs = [conv(output) for conv in self.conv1B1]
        outputs[1] = self.conv3B2[0](outputs[1])
        outputs[2] = self.conv3B3[1](outputs[2])
        outputs[2] = self.conv3B4[0](outputs[2])
        output = torch.cat(outputs, dim=1)

        outputs = [conv(output) for conv in self.conv1C1]
        outputs[1] = self.conv3C2[0](outputs[1])
        outputs[2] = self.conv3C3[1](outputs[2])
        outputs[2] = self.conv3C4[0](outputs[2])
        output = torch.cat(outputs, dim=1)

        ## 256
        output = self.conv11(output)

        if self.channel_last:
            output = output.transpose(1, 2)
        output = self.dense(output)

        return output

class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        while (((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if (encoder_output):
                state, _ = fn((state, encoder_output))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = (
                        (state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step += 1
        return previous_state, (remainders, n_updates)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).float()


class lightning(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(42, 128, 11, padding=5, bias=True),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        encoder_layer1 = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=1)

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 64, 11, padding=5, bias=True),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        encoder_layer2 = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=1)

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, dataset.num_classes, 11, padding=5),
            nn.BatchNorm1d(dataset.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        input = input.transpose(1,2)
        output = self.conv1(input)
        output = output.transpose(1, 2)
        output = self.transformer_encoder1(output)
        output = output.transpose(1, 2)
        output = self.conv2(output)
        output = output.transpose(1, 2)
        output = self.transformer_encoder2(output)
        output = output.transpose(1, 2)
        output = self.conv3(output)
        return output


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(42, 42)
        self.inception_base = Inception(True)
        self.lightning = lightning()
        self.final_dense = torch.nn.Sequential(nn.Linear(42, 8), nn.Softmax(dim=-1))
        self.fusion = Fusion(8)

    def forward(self, X_one_hot, X_pssm):
        """
        :param X_one_hot: batch_size, seq_len , 22
        :param X_pssm:
        :param X_adj_in: batch_size, 22, 22
        :param X_adj_out: batch_size, 22, 22
        :param X_lengths:
        :return:
        """
        input = torch.cat([X_one_hot, X_pssm], dim=2)
        output = self.input_layer(input)
        # for i in range(3):
        # output1
        output1 = self.lightning(output)

        # outpu2
        output2 = self.inception_base(output)
        ### x: batch, seq_len, 42

        output2 = self.final_dense(output2)

        output1 = output1.transpose(1,2)

        return self.fusion(output1,output2)

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR, weight_decay=0.0005)

    def training_step(self, batch, batch_idx):
        X_one_hot, X_pssm, X_lengths, Y = batch
        logits = self(X_one_hot, X_pssm)
        _, Y = Y.max(dim=2)
        loss = compute_loss(logits, Y, X_lengths)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        X_one_hot, X_pssm, X_lengths, Y = batch
        logits = self(X_one_hot, X_pssm)
        _, Y = Y.max(dim=2)
        loss = compute_loss(logits, Y, X_lengths)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(avg_loss)
        return {'val_loss': avg_loss}

    def Q8_accuracy(self, y_pred, y_true):
        pred = y_pred.cpu().numpy()
        real = y_true.cpu().numpy()
        total = real.shape[0] * real.shape[1]
        correct = 0
        for i in range(real.shape[0]):  # per element in the batch
            for j in range(real.shape[1]):  # per aminoacid residue
                if np.sum(real[i, j, :]) == 0:  # real[i, j, dataset.num_classes - 1] > 0 # if it is padding
                    total = total - 1
                else:
                    if real[i, j, np.argmax(pred[i, j, :])] > 0:
                        correct = correct + 1

        return correct, total

    def test_step(self, batch, batch_idx):
        X_one_hot, X_pssm, X_lengths, Y = batch
        y = Y
        logits = self(X_one_hot, X_pssm)
        _, Y = Y.max(dim=2)
        loss = compute_loss(logits, Y, X_lengths)
        correct, total = self.Q8_accuracy(logits, y)
        return {'test_loss': loss, 'correct': correct, 'total': total}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        corrects = sum([x['correct'] for x in outputs])
        totals = sum([x['total'] for x in outputs])
        acc = corrects / totals
        return {'avg_test_loss': avg_loss, 'Q8_accuracy': acc}


checkpoint_callback = ModelCheckpoint('logs/')
early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=1000, verbose=False, mode='min')

from pytorch_lightning import Trainer

model = Model()
trainer = Trainer(gpus=1, num_nodes=1, terminate_on_nan=True, checkpoint_callback=checkpoint_callback,
                  early_stop_callback=early_stop_callback)
trainer.fit(model)
# model = Model.load_from_checkpoint('logs/epoch=212.ckpt')
trainer.test()
