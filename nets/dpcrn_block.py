import torch
import torch.nn as nn
import os
import sys
from show import show_params, show_model
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT
from complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


class InstantLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size)
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim1, dim2, eps=1e-8, elementwise_affine=True):
        super(InstantLayerNorm, self).__init__()
        self.dim1 = dim1        
        self.dim2 = dim2
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim1, self.dim2, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim1, self.dim2, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        #print(x.shape)

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x

class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        #input = input.to(device)
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output

class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, output_size, freq_dim, down_sample,
                 dropout=0, num_layers=1, bidirectional=False):

        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.freq_dim = freq_dim

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout,
                                          bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_norm.append(InstantLayerNorm(input_size, self.freq_dim//(2**down_sample), elementwise_affine=True))
            # default is to use noncausal LayerNorm for inter-chunk RNN. For causal setting change it to causal normalization techniques accordingly.
            self.col_norm.append(InstantLayerNorm(input_size, self.freq_dim//(2**down_sample), elementwise_affine=True))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size, output_size, 1)
                                    )

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        #input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape

        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
            #print(row_input.shape)
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2,
                                                                             1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
            #print(col_input.shape)
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1,
                                                                             2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output
        
        output = self.output(output) # B, output_size, dim1, dim2

        return output

class DPCRN(nn.Module):

    def __init__(
                    self,
                    rnn_layers=2,
                    rnn_units=64,
                    use_clstm=False,
                    use_cbn = False,
                    kernel_size=5,
                    down_sample=2,
                    feat_dim=256,
                    input_channel_rate=1,
                    kernel_num=[64,64,64],
                ):
        ''' 
            
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag

        '''

        super(DPCRN, self).__init__()

        self.rnn_units = rnn_units
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        self.kernel_num = [2*input_channel_rate]+kernel_num
        self.use_clstm = use_clstm
        self.stride = (2, 1)
        self.padding = (2, 0)
        self.output_padding = (0, 0)
        self.down_sample =  down_sample
        self.feat_dim = feat_dim

        #bidirectional=True
        bidirectional=False
        fac = 2 if bidirectional else 1


        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num)-1):
            if idx > 0:
                self.kernel_size = 3
                self.padding = (1, 0)
            if idx > self.down_sample - 1 :
                self.stride = (1, 1)
            self.encoder.append(
                nn.Sequential(
                    nn.ConstantPad2d([1, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx+1],
                        kernel_size=(self.kernel_size, 2),
                        stride=self.stride,
                        padding=self.padding
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx+1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx+1]),
                    nn.PReLU()
                )
            )


        self.enhance = DPRNN(
                    rnn_type = 'LSTM',
                    input_size= self.kernel_num[-1],
                    hidden_size=self.rnn_units,
                    output_size=self.kernel_num[-1],
                    num_layers=2,
                    freq_dim=self.feat_dim,
                    down_sample = self.down_sample,
                    dropout=0.0,
                    bidirectional=bidirectional,
            )


        for idx in range(len(self.kernel_num)-1, 0, -1):
            if idx != 1:
                if idx < self.down_sample + 1 :
                    self.stride = (2, 1)
                    self.output_padding = (1, 0)
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],

                        kernel_size =(self.kernel_size, 2),
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=self.output_padding
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx-1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx-1]),
                    #nn.ELU()
                    nn.PReLU(),
                    )
                )
            else:
                self.kernel_size = 5
                self.padding = (2, 0)
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1]//input_channel_rate,
                        kernel_size =(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=self.padding,
                        output_padding=self.output_padding
                    ),
                    )
                )

        #self.flatten_parameters() 

    '''
    def flatten_parameters(self): 
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()
    '''

    def forward(self, inputs, lens=None):

        out = inputs

        encoder_out = []
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #print('encoder', out.size())
            encoder_out.append(out)
        
        batch_size, channels, dims, lengths = out.size()
        out = self.enhance(out)

        for idx in range(len(self.decoder)):
            out = complex_cat([out,encoder_out[-1 - idx]],1)
            out = self.decoder[idx](out)
            out = out[...,:-1]
            #print('decoder', out.size())

        return out