import numpy as np
import argparse
import os


import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader
from .lstm_encoder import LSTMStackedEncoders
from .lstm_decoder import LSTMStackedDecoders


class LSTMLadder(torch.nn.Module):
    def __init__(self, n_times, n_classes, encoder_sizes, decoder_sizes,
                 encoder_activations, encoder_train_bn_scalings, noise_std, use_cuda):
        super(LSTMLadder, self).__init__()
        self.use_cuda = use_cuda
        decoder_in = encoder_sizes[-1]
        encoder_in = decoder_sizes[-1]
        self.se = LSTMStackedEncoders(n_times, encoder_in, encoder_sizes, encoder_activations,
                                        encoder_train_bn_scalings, noise_std, use_cuda)
        self.de = LSTMStackedDecoders(n_times, decoder_in, decoder_sizes, encoder_in, use_cuda)

    def forward_encoders_clean(self, data):
        return self.se.forward_clean(data)

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_encoders_clean_predict(self, data):
        h = self.se.forward_clean(data)
        h = h.mean(dim=1)
        return h

    def forward_encoders_noise_predict(self, data):
        h = self.se.forward_noise(data)
        h = h.mean(dim=1)
        return h

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        return self.de.forward(tilde_z_layers, encoder_output, tilde_z_bottom)

    def get_encoders_tilde_z(self, reverse=True):
        return self.se.get_encoders_tilde_z(reverse)

    def get_encoders_z_pre(self, reverse=True):
        return self.se.get_encoders_z_pre(reverse)

    def get_encoder_tilde_z_bottom(self):
        return self.se.buffer_tilde_z_bottom.clone()

    def get_encoders_z(self, reverse=True):
        return self.se.get_encoders_z(reverse)

    def get_decoder_bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        return self.de.bn_hat_z_layers(hat_z_layers, z_pre_layers)
    
    def get_decoder_mu_layers(self):
        return self.de.get_mu_layers()

    def get_decoder_v_layers(self):
        return self.de.get_v_layers()
