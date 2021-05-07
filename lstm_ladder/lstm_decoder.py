import torch
from torch import nn
from torch.nn import Parameter


class LSTMDecoder(nn.Module):
    def __init__(self, n_times, d_in, d_out, use_cuda):
        super(LSTMDecoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_cuda = use_cuda

        # g function parameters
        if self.use_cuda:
            self.a1 = Parameter(torch.zeros(1, n_times, d_in).cuda(), requires_grad=True)
            self.a2 = Parameter(torch.ones(1, n_times, d_in).cuda(), requires_grad=True)
            self.a3 = Parameter(torch.zeros(1, n_times, d_in).cuda(), requires_grad=True)
            self.a4 = Parameter(torch.zeros(1, n_times, d_in).cuda(), requires_grad=True)
            self.a5 = Parameter(torch.zeros(1, n_times, d_in).cuda(), requires_grad=True)

            self.a6 = Parameter(torch.zeros(1, n_times, d_in).cuda(), requires_grad=True)
            self.a7 = Parameter(torch.ones(1, n_times, d_in).cuda(), requires_grad=True)
            self.a8 = Parameter(torch.zeros(1, n_times, d_in).cuda(), requires_grad=True)
            self.a9 = Parameter(torch.zeros(1, n_times, d_in).cuda(), requires_grad=True)
            self.a10 = Parameter(-1 * torch.ones(1, n_times, d_in).cuda(), requires_grad=True)
        else:
            self.a1 = Parameter(torch.zeros(1, n_times, d_in), requires_grad=True)
            self.a2 = Parameter(torch.ones(1, n_times, d_in), requires_grad=True)
            self.a3 = Parameter(torch.zeros(1, n_times, d_in), requires_grad=True)
            self.a4 = Parameter(torch.zeros(1, n_times, d_in), requires_grad=True)
            self.a5 = Parameter(torch.zeros(1, n_times, d_in), requires_grad=True)

            self.a6 = Parameter(torch.zeros(1, n_times, d_in), requires_grad=True)
            self.a7 = Parameter(torch.ones(1, n_times, d_in), requires_grad=True)
            self.a8 = Parameter(torch.zeros(1, n_times, d_in), requires_grad=True)
            self.a9 = Parameter(torch.zeros(1, n_times, d_in), requires_grad=True)
            self.a10 = Parameter(-1 * torch.zeros(1, n_times, d_in), requires_grad=True)
        
        # LSTM MODULE
        if self.d_out is not None:
            self.lstm = torch.nn.LSTM(d_in, d_out, batch_first=True)
            if self.use_cuda:
                try:
                    self.lstm.cuda()
                except:
                    self.lstm.cuda()

        # BN MODULE
            self.bn = torch.nn.BatchNorm1d(n_times, affine=False)
            if self.use_cuda:
                self.bn.cuda()

        # BUFFER
        self.buffer_hat_z = None
        self.buffer_mu = None
        self.buffer_v = None

    # g function for calculating hat_z
    def g(self, tilde_z, u):
        if self.use_cuda:
            ones = torch.ones(tilde_z.size()[0], 1, 1).cuda()
        else:
            ones = torch.ones(tilde_z.size()[0], 1, 1)
        b_a1 = ones*self.a1
        b_a2 = ones*self.a2
        b_a3 = ones*self.a3
        b_a4 = ones*self.a4
        b_a5 = ones*self.a5
        b_a6 = ones*self.a6
        b_a7 = ones*self.a7
        b_a8 = ones*self.a8
        b_a9 = ones*self.a9
        b_a10 = ones*self.a10

        mu = torch.mul(b_a1, torch.sigmoid(torch.mul(b_a2, u) + b_a3)) + torch.mul(b_a4, u) + b_a5
        v = torch.sigmoid(torch.mul(b_a6, torch.sigmoid(torch.mul(b_a7, u) + b_a8)) + torch.mul(b_a9, u) + b_a10)

        self.buffer_mu = mu.detach().clone()
        self.buffer_v = v.detach().clone()

        hat_z = torch.mul(mu - tilde_z, v) + tilde_z
        return hat_z

    def forward(self, tilde_z, u):
        # g function for hat_z
        hat_z = self.g(tilde_z, u)
        # store hat_z in buffer
        self.buffer_hat_z = hat_z
        if self.d_out is not None:
            t = self.lstm.forward(hat_z)[0]
            u_below = self.bn(t)
            return u_below
        else:
            return None


class LSTMStackedDecoders(nn.Module):
    def __init__(self, n_times, d_in, d_decoders, origin_size, use_cuda):
        super(LSTMStackedDecoders, self).__init__()
        self.use_cuda = use_cuda
        
        # TOP BN PART
        self.bn_u_top = torch.nn.BatchNorm1d(n_times, affine=False)
        if self.use_cuda:
            self.bn_u_top.cuda()

        # DECODERS
        self.decoders_ref = []
        self.decoders = torch.nn.Sequential()
        n_decoders = len(d_decoders)
        for i in range(n_decoders):
            if i == 0:
                d_input = d_in
            else:
                d_input = d_decoders[i - 1]
            d_output = d_decoders[i]
            decoder_ref = "decoder_" + str(i)
            decoder = LSTMDecoder(n_times, d_input, d_output, use_cuda)
            self.decoders_ref.append(decoder_ref)
            self.decoders.add_module(decoder_ref, decoder)
        self.bottom_decoder = LSTMDecoder(n_times, origin_size, None, use_cuda)

    def forward(self, tilde_z_layers, u_top, tilde_z_bottom):
        hat_z = []
        u = self.bn_u_top(u_top)
        for i in range(len(self.decoders_ref)):
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            tilde_z = tilde_z_layers[i]
            u = decoder.forward(tilde_z, u)
            hat_z.append(decoder.buffer_hat_z)
        self.bottom_decoder.forward(tilde_z_bottom, u)
        hat_z_bottom = self.bottom_decoder.buffer_hat_z.clone()
        hat_z.append(hat_z_bottom)
        return hat_z

    def bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        assert len(hat_z_layers) == len(z_pre_layers)
        hat_z_layers_normalized = []
        for i, (hat_z, z_pre) in enumerate(zip(hat_z_layers, z_pre_layers)):
            ones = torch.ones(z_pre.size()[0], 1, 1)
            mean = torch.mean(z_pre, dim=0, keepdim=True)
            std = torch.std(z_pre, dim=0, keepdim=True)
            if self.use_cuda:
                hat_z = hat_z.cuda()
                ones = ones.cuda()
                mean = mean.cuda()
                std = std.cuda()
            hat_z_normalized = torch.div(hat_z - ones * mean, ones * std)
            hat_z_layers_normalized.append(hat_z_normalized)
        return hat_z_layers

    def get_mu_layers(self): 
        mu_layers = []
        for i in range(len(self.decoders_ref)):
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            mu_layers.append(decoder.buffer_mu)
        mu_layers.append(self.bottom_decoder.buffer_mu)
        return mu_layers

    def get_v_layers(self): 
        v_layers = []
        for i in range(len(self.decoders_ref)):
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            v_layers.append(decoder.buffer_v)
        v_layers.append(self.bottom_decoder.buffer_v)
        return v_layers
