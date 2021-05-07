import numpy as np
import torch
from torch import nn


class LINEAREncoder(nn.Module):
    def __init__(self, d_in, d_out, activation_type,
                 train_bn_scaling, noise_level, use_cuda):
        super(LINEAREncoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation_type = activation_type
        self.train_bn_scaling = train_bn_scaling
        self.noise_level = noise_level
        self.use_cuda = use_cuda
        
        # LINEAR MODULE
        self.linear = nn.Linear(d_in, d_out)
        if self.use_cuda:
            try:
                self.linear.cuda()
            except:
                self.linear.cuda()
        
        # BN MODULE
        # clean and noisy tensors flow in separate ways
        # gaussian noise will be added after BN in noisy way
        # For ReLU, both Gamma and Beta are trained
        # For Softmax, both Gamma and Beta are trained
        self.bn_clean = nn.BatchNorm1d(d_out, affine=False)
        self.bn_noise = nn.BatchNorm1d(d_out, affine=False)
        
        if self.use_cuda:
            self.bn_beta = torch.zeros(d_out).cuda()
        else:
            self.bn_beta = torch.zeros(d_out)
        
        if self.train_bn_scaling:
            if self.use_cuda:
                self.bn_gamma = torch.ones(d_out).cuda()
            else:
                self.bn_gamma = torch.ones(d_out)

        # ACTIVATION MODULE
        if activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'leakyrelu':
            self.activation = torch.nn.LeakyReLU()
        elif activation_type == 'softmax':
            self.activation = torch.nn.Softmax(dim=-1)
        else:
            raise ValueError("invalid Acitvation type")

        # BUFFER
        self.buffer_z_pre = None
        self.buffer_z = None
        self.buffer_tilde_z = None

    # implement Gamma and Beta after BN
    def bn_gamma_beta(self, x):
        if self.train_bn_scaling:
            h = self.bn_gamma * (x + self.bn_beta)
        else:
            h = x + self.bn_beta
        return h

    # clean tensor way
    def forward_clean(self, h):
        z_pre = self.linear(h)
        self.buffer_z_pre = z_pre.detach().clone()
        z = self.bn_clean(z_pre)
        self.buffer_z = z.detach().clone()
        z_gb = self.bn_gamma_beta(z)
        h = self.activation(z_gb)
        return h

    # noisy tensor way
    def forward_noise(self, tilde_h):
        z_pre = self.linear(tilde_h)
        z = self.bn_noise(z_pre)
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=z.size())
        if self.use_cuda:
            noise = torch.FloatTensor(noise).cuda()
        else:
            noise = torch.FloatTensor(noise)
        tilde_z = z + noise
        self.buffer_tilde_z = tilde_z
        z = self.bn_gamma_beta(tilde_z)
        h = self.activation(z)
        return h


class LINEARStackedEncoders(nn.Module):
    def __init__(self, d_in, d_encoders, activation_types,
                 train_bn_scalings, noise_std, use_cuda):
        super(LINEARStackedEncoders, self).__init__()
        self.noise_std = noise_std
        self.use_cuda = use_cuda

        # BOTTOM PART
        self.bn_bottom = nn.BatchNorm1d(d_in, affine=False)
        
        # ENCODERS
        self.encoders_ref = []
        self.encoders = torch.nn.Sequential()
        n_encoders = len(d_encoders)
        for i in range(n_encoders):
            if i == 0:
                d_input = d_in
            else:
                d_input = d_encoders[i - 1]
            d_output = d_encoders[i]
            activation = activation_types[i]
            train_bn_scaling = train_bn_scalings[i]
            encoder_ref = "encoder_" + str(i)
            encoder = LINEAREncoder(d_input, d_output, activation,
                                    train_bn_scaling, noise_std, use_cuda)
            self.encoders_ref.append(encoder_ref)
            self.encoders.add_module(encoder_ref, encoder)
        
        # BUFFER
        self.buffer_tilde_z_bottom = None
        self.buffer_z_pre_bottom = None
        self.buffer_z_bottom = None

    # clean tensor way
    def forward_clean(self, x):
        h = x
        self.buffer_z_pre_bottom = h.clone()
        self.buffer_z_bottom = self.bn_bottom(h)
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_clean(h)
        return h

    # noisy tensor way
    def forward_noise(self, x):
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=x.size())
        if self.use_cuda:
            noise = torch.FloatTensor(noise).cuda()
        else:
            noise = torch.FloatTensor(noise)
        h = x + noise
        self.buffer_tilde_z_bottom = h.clone()
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_noise(h)
        return h

    def get_encoders_tilde_z(self, reverse=True):
        tilde_z_layers = []
        tilde_z_layers.append(self.buffer_tilde_z_bottom)
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            tilde_z = encoder.buffer_tilde_z.clone()
            tilde_z_layers.append(tilde_z)
        if reverse:
            tilde_z_layers.reverse()
        return tilde_z_layers

    def get_encoders_z_pre(self, reverse=True):
        z_pre_layers = []
        z_pre_layers.append(self.buffer_z_pre_bottom)
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z_pre = encoder.buffer_z_pre.clone()
            z_pre_layers.append(z_pre)
        if reverse:
            z_pre_layers.reverse()
        return z_pre_layers

    def get_encoders_z(self, reverse=True):
        z_layers = []
        z_layers.append(self.buffer_z_bottom)
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z = encoder.buffer_z.clone()
            z_layers.append(z)
        if reverse:
            z_layers.reverse()
        return z_layers
