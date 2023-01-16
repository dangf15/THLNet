import torch
import torch.nn as nn
import torch
import copy
import numpy as np
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
from utils import InstantLayerNorm2d, InstantLayerNorm1d


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()


        #self.self_attn = PerformerSelfAttention(d_model, nhead)
        # Implementation of Feedforward model
        #dim_feedforward = d_model * 2
        #self.linear1 = Linear(d_model, dim_feedforward)
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        #self.linear2 = Linear(dim_feedforward, d_model)

        if bidirectional:
            self.linear2 = Linear(d_model*2*2, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm1 = InstantLayerNorm1d(d_model)
            self.dropout1 = Dropout(dropout)
        else:
            self.linear2 = Linear(d_model*2, d_model)

        self.norm2 = InstantLayerNorm1d(d_model)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.bidirectional = bidirectional

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
       Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        if self.bidirectional:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)

            src = self.norm1(src)
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



class Dual_Transformer(nn.Module):
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
        flag: if 'casual' flag = 1, else flag = 0 (default=0).
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(Dual_Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(self.input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )
        
        # dual-path RNN
        self.freq_trans = nn.ModuleList([])
        self.time_trans = nn.ModuleList([])
        self.freq_norm = nn.ModuleList([])
        self.time_norm = nn.ModuleList([])

        for i in range(num_layers):
            self.freq_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True))
            self.time_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=False))
            self.freq_norm.append(InstantLayerNorm2d(input_size//2))
            self.time_norm.append(InstantLayerNorm2d(input_size//2))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size//2, self.output_size, 1)
                                    )


    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, t, f]
        b, c, t, f = input.shape
        output = self.input(input)
        for i in range(len(self.freq_trans)):

            time_input = output.permute(2, 0, 3, 1).contiguous().view(t, b*f, -1)
            time_output = self.time_trans[i](time_input)
            time_output = time_output.view(t, b, f, -1).permute(1, 3, 0, 2).contiguous()
            time_output = self.time_norm[i](time_output)
            output = output + time_output


            freq_input = output.permute(3, 0, 2, 1).contiguous().view(f, b*t, -1)
            freq_output = self.freq_trans[i](freq_input)
            freq_output = freq_output.view(f, b, t, -1).permute(1, 3, 2, 0).contiguous()
            freq_output = self.freq_norm[i](freq_output)
            output = output + freq_output


        del freq_input, freq_output, time_input, time_output

        output = self.output(output)
        return output


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


class DenseBlock(nn.Module):
    def __init__(self, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            #print(pad_length)
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length , 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), InstantLayerNorm2d(self.in_channels))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
        return out

class DFNet(nn.Module):
    def __init__(self, width=64, input_channel_rate=1, output_channel=2):

        super(DFNet, self).__init__()
        # self.device = device
        self.in_channels = 2 * input_channel_rate
        self.out_channels = output_channel
        self.kernel_size = (2, 3)
        # self.elu = nn.SELU(inplace=True)
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.width = width

        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 256]
        self.inp_norm = InstantLayerNorm2d(width)
        self.inp_prelu = nn.PReLU(self.width)

        self.enc_dense1 = DenseBlock(4, self.width)
        self.dual_transformer = Dual_Transformer(self.width, self.width, num_layers=4)  # # [b, 64, nframes, 8]

        # gated output layer
        self.output1 = nn.Sequential(
            nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1),
            nn.Tanh()
        )
        self.output2 = nn.Sequential(
            nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1),
            nn.Sigmoid()
        )

        self.dec_dense1 = DenseBlock(4, self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))


    def forward(self, x):

        x = x.permute(0,1,3,2) # [B, 2, num_frames, num_bins]
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, num_frames, frame_size]
        out = self.enc_dense1(out)   # [b, 64, num_frames, frame_size]

        out = self.dual_transformer(out)  # [b, 64, num_frames, 256]
        out = self.output1(out) * self.output2(out)  # mask [b, 64, num_frames, 256]
        out = self.dec_dense1(out)
        out = self.out_conv(out)
        out = out.permute(0,1,3,2)
        return out