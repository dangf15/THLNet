import torch
import torch.nn as nn
import os
import sys
from show import show_params, show_model
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT
from nets.dfnet_block import DFNet
from nets.dpcrn_block import DPCRN
from complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm


class ds_block(nn.Module):

    def __init__(
                    self,
                    width=48,
                    win_len=512,
                    win_inc=256,
                    fft_len=512,
                    win_type='hanning',
                    masking_mode='E',
                ):


        super(ds_block, self).__init__()

        # for fft 
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.masking_mode = masking_mode
        self.freq_dim = 512
        self.ds_dim = 64
        self.groups = self.ds_dim // 2

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        #self.cln = InstantLayerNorm(2, 256, elementwise_affine=True)
        self.ana_conv = ComplexConv2d(self.freq_dim, self.ds_dim, (1,1), (1,1), (0,0), (1,1), self.groups)
        self.sys_conv = ComplexConv2d(self.ds_dim, self.freq_dim, (1,1), (1,1), (0,0), (1,1), self.groups)
        #self.ana_conv = ComplexConv2d(self.freq_dim, self.ds_dim, (1,1), (1,1), (0,0), (1,1))
        #self.sys_conv = ComplexConv2d(self.ds_dim, self.freq_dim, (1,1), (1,1), (0,0), (1,1))

        self.enh_block = DFNet(width=width)
        #self.enh_block = DPCRN(feat_dim=self.ds_dim//2)



    def forward(self, inputs, lens=None):
        specs = self.stft(inputs)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase


        cspecs = torch.cat([real[:,1:,:],imag[:,1:,:]],1)
        #print(cspecs.unsqueeze(-1).shape)
        
        out = self.ana_conv(cspecs.unsqueeze(-1)).squeeze(-1)#self.cln(cspecs)
        real, imag = torch.chunk(out, 2, 1)
        out = torch.stack([real, imag],1)

        out = self.enh_block(out)
        real = out[:,0,:,:]
        imag = out[:,1,:,:]
        out = torch.cat([real,imag],1)
        out = self.sys_conv(out.unsqueeze(-1)).squeeze(-1)
        mask_real, mask_imag = torch.chunk(out, 2, 1)
        mask_real = F.pad(mask_real, [0,0,1,0])
        mask_imag = F.pad(mask_imag, [0,0,1,0])
        mask_mags = (mask_real**2+mask_imag**2)**0.5
        real_phase = mask_real/(mask_mags+1e-8)
        imag_phase = mask_imag/(mask_mags+1e-8)
        mask_phase = torch.atan2(
                            imag_phase,
                            real_phase
                        )

        #mask_mags = torch.clamp_(mask_mags,0,100) 
        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags*spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags*torch.cos(est_phase)
        imag = est_mags*torch.sin(est_phase)

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)

        out_wav = torch.squeeze(out_wav, 1)
        #out_wav = torch.tanh(out_wav)
        out_wav = torch.clamp_(out_wav,-1,1)
        return out_spec,  out_wav


class re_block(nn.Module):

    def __init__(
                    self,
                    width=48,
                    win_len=512,
                    win_inc=256,
                    fft_len=512,
                    win_type='hanning',
                    masking_mode='C',
                ):

        super(re_block, self).__init__()

        # for fft 
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.freq_dim = 128
        self.masking_mode = masking_mode

        fix=True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        #self.cln = InstantLayerNorm(2, 256, elementwise_affine=True)

        #self.enh_block = DFNet(width=width,input_channel_rate=2)    

        self.enh_block = DPCRN(feat_dim=self.freq_dim,input_channel_rate=2)

    def forward(self, x1, x2,lens=None):

        specs1 = self.stft(x1)
        real1 = specs1[:,:self.fft_len//2+1]
        imag1 = specs1[:,self.fft_len//2+1:]
        spec_mags1 = torch.sqrt(real1**2+imag1**2+1e-8)
        spec_phase1 = torch.atan2(imag1, real1)

        specs2 = self.stft(x2)
        real2 = specs2[:,:self.fft_len//2+1]
        imag2 = specs2[:,self.fft_len//2+1:]
        spec_mags2 = torch.sqrt(real2**2+imag2**2+1e-8)
        spec_phase2 = torch.atan2(imag2, real2)

        real = torch.stack([real1,real2],1)
        imag = torch.stack([imag1,imag2],1)
        cspecs = torch.cat([real,imag],1)
        cspecs = cspecs[:,:,1:self.freq_dim+1]


        out = cspecs#self.cln(cspecs)
        out = self.enh_block(out)

        mask_real = out[:,0]
        mask_imag = out[:,1]
        mask_real = F.pad(mask_real, [0,0,1,0])
        mask_imag = F.pad(mask_imag, [0,0,1,0])

        spec_mags2 = spec_mags2[:,:self.freq_dim+1,:]
        spec_phase2 = spec_phase2[:,:self.freq_dim+1,:]
        real2 = real2[:,:self.freq_dim+1,:]
        imag2 = imag2[:,:self.freq_dim+1,:]

        if self.masking_mode == 'E' :
            mask_mags = (mask_real**2+mask_imag**2)**0.5
            real_phase = mask_real/(mask_mags+1e-8)
            imag_phase = mask_imag/(mask_mags+1e-8)
            mask_phase = torch.atan2(
                            imag_phase,
                            real_phase
                        )

            #mask_mags = torch.clamp_(mask_mags,0,100) 
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags*spec_mags2  
            est_phase = spec_phase2 + mask_phase
            real = est_mags*torch.cos(est_phase)
            imag = est_mags*torch.sin(est_phase)
        elif self.masking_mode == 'C':
            real,imag = real2*mask_real-imag2*mask_imag + real1[:,:self.freq_dim+1,:], real2*mask_imag+imag2*mask_real + imag1[:,:self.freq_dim+1,:]

        real = torch.cat([real, real1[:,self.freq_dim+1:,:]],1)
        imag = torch.cat([imag, imag1[:,self.freq_dim+1:,:]],1)
        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)

        out_wav = torch.squeeze(out_wav, 1)
        #out_wav = torch.tanh(out_wav)
        out_wav = torch.clamp_(out_wav,-1,1)
        return out_spec,  out_wav


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.block1 = ds_block(masking_mode='E')
        self.block2 = re_block(masking_mode='C')

        show_model(self)
        show_params(self)

    def forward(self, x):

            wav_lst = []
            out1 = self.block1(x)[1]
            wav_lst.append(out1)  
            out2 = self.block2(out1, x)[1]
            wav_lst.append(out2)
            #print(out1.shape)

            return wav_lst

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    from thop import profile
    torch.manual_seed(10)
    torch.autograd.set_detect_anomaly(True)
    inputs = torch.randn([1,1,16000]).clamp_(-1,1).cuda()
    labels = torch.randn([1,16000]).clamp_(-1,1).cuda()

    net = Net().cuda()
    macs, params = profile(net, inputs=(inputs ),)
    print('mac:',macs/1e9,'\nparams:',params)
#    print('Number of learnable parameters: %d' % numParams(net))                                                   