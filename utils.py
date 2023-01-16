from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import os
import shutil


class Checkpoint(object):
    def __init__(self, start_epoch=None, start_iter=None, train_loss=None, eval_loss=None, best_val_loss=float("inf"),
                 prev_val_loss=float("inf"), state_dict=None, optimizer=None, num_no_improv=0, half_lr=False):
        self.start_epoch = start_epoch
        self.start_iter = start_iter
        self.train_loss = train_loss
        self.eval_loss = eval_loss

        self.best_val_loss = best_val_loss
        self.prev_val_loss = prev_val_loss

        self.state_dict = state_dict
        self.optimizer = optimizer

        self.num_no_improv = num_no_improv
        self.half_lr = half_lr


    def save(self, is_best, filename, best_model):
        print('Saving checkpoint at "%s"' % filename)
        torch.save(self, filename)
        if is_best:
            print('Saving the best model at "%s"' % best_model)
            shutil.copyfile(filename, best_model)
        print('\n')


    def load(self, filename):
        # filename : model path
        if os.path.isfile(filename):
            print('Loading checkpoint from "%s"\n' % filename)
            checkpoint = torch.load(filename, map_location='cpu')

            self.start_epoch = checkpoint.start_epoch
            self.start_iter = checkpoint.start_iter
            self.train_loss = checkpoint.train_loss
            self.eval_loss = checkpoint.eval_loss

            self.best_val_loss = checkpoint.best_val_loss
            self.prev_val_loss = checkpoint.prev_val_loss
            self.state_dict = checkpoint.state_dict
            self.optimizer = checkpoint.optimizer
            self.num_no_improv = checkpoint.num_no_improv
            self.half_lr = checkpoint.half_lr
        else:
            raise ValueError('No checkpoint found at "%s"' % filename)

class InstantLayerNorm1d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(InstantLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, 1, num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, 1, num_features), requires_grad=False)
            self.bias = Variable(torch.zeros(1, 1, num_features), requires_gra=False)

    def forward(self, inpt):
        # inpt: (T,B,C)
        seq_len, b_size, channel = inpt.shape
        ins_mean = torch.mean(inpt, dim=-1, keepdim=True)  # (T,B,1)
        ins_std = (torch.var(inpt, dim=-1, keepdim=True) + self.eps).pow(0.5)  # (T,B,1)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class InstantLayerNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(InstantLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1, 1), requires_grad=False)

    def forward(self, inpt):
        # inpt: (B,C,T,F)
        ins_mean = torch.mean(inpt, dim=[1,3], keepdim=True)  # (B,C,T,1)
        ins_std = (torch.std(inpt, dim=[1,3], keepdim=True) + self.eps).pow(0.5)  # (B,C,T,1)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())                     



def checkcausal_net(net):
    """
    The whole pipeline give a latency of aec_shift + net win = 48ms
    """
    net_causal = net()
    net_causal = net_causal.eval()
    d = torch.device('cpu')
    noisy_wavs = torch.randn([1,1,16000]).clamp_(-1,1)
    # net_causal, net_noncal = net_causal.to(d).eval(), net_noncal.to(d).eval()
    # noisy_wavs =  noisy_wavs.to(d)
    # noncausal model uses utt-level info
    noisy_wavs[0,0,-1] = np.nan
    out = net_causal(noisy_wavs)

    # assert torch.isnan(out[0,0,0]) or torch.isinf(out[0,0,0])
    '''
    with torch.no_grad():
        out1 = net_causal(noisy_wavs)[0].squeeze()
        for i in range(512*16,512*18,fs):
            noisy_wavs2 = noisy_wavs.clone()
            noisy_wavs2[0,2,i:] = 1000 + torch.rand_like(noisy_wavs2[0,2,i:])
            out2 = net_causal(noisy_wavs2)[0].squeeze()
            print((i-((out1-out2).abs()>1e-8).float().argmax())/fs)
            print((i-((out1-out2).abs()>1e-8).float().argmax())/fs)
        for i in range(512*16,512*18,fs):
            noisy_wavs2 = noisy_wavs.clone()
            noisy_wavs2[0,1,i:] = 1000 + torch.rand_like(noisy_wavs2[0,1,i:])
            out2 = net_causal(noisy_wavs2)[0].squeeze()
            print((i-((out1-out2).abs()>1e-8).float().argmax())/fs)
            print((i-((out1-out2).abs()>1e-8).float().argmax())/fs)
    '''