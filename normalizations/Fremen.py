import torch
import torch.nn as nn
import numpy as np
torch.autograd.set_detect_anomaly(True)


class Norm(nn.Module):
    def __init__(self, configs, eps=1e-5):
        super(Norm, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.device = configs.device
        self.hidden = 256
        self.eps = eps

    def forward(self, x):
        return self._reweight(x)

    def _build_model(self):
        # MLP for the real part of frequency components in the input data
        self.linear_r = nn.Sequential(
            nn.Linear(self.weight.size(0), self.hidden),  # bottleneck structure
            nn.RReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden, self.weight.size(0)),
        )
        # MLP for the imaginary part of frequency components in the input data
        self.linear_i = nn.Sequential(
            nn.Linear(self.weight.size(0), self.hidden),  # bottleneck structure
            nn.RReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden, self.weight.size(0)),
        )

    def _weight_initialization(self, train_loader):
        num_channels = self.enc_in
        freq_length = self.seq_len

        # calculate empirical standard deviations
        amplitude_sum = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        amplitude_squared_sum = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        count = 0
        with torch.no_grad():
            for data in train_loader:
                lookback_window = data[0].float().to(self.device)
                xf = torch.fft.fft(lookback_window, dim=1)
                amplitude = torch.abs(xf)
                amplitude_sum += amplitude.sum(dim=0)
                amplitude_squared_sum += (amplitude ** 2).sum(dim=0)
                count += lookback_window.size(0)
        mean_amplitude = amplitude_sum / count
        mean_amplitude_squared = amplitude_squared_sum / count
        variance_amplitude = mean_amplitude_squared - (mean_amplitude ** 2)
        std_amplitude = torch.sqrt(variance_amplitude + self.eps)

        # calculate RBF kernel eigenvalues
        w = torch.fft.fftfreq(self.seq_len).reshape(-1, 1).expand(self.seq_len, self.enc_in).to(self.device)
        sigma = std_amplitude.to(self.device)
        self.weight = sigma * torch.exp(-(sigma * w) ** 2 / 2) * np.sqrt(2 * np.pi)
        self._build_model()

    def _reweight(self, input):
        # apply fourier transform to the input data
        data_fft = torch.fft.fft(input, dim=1)

        # learn eigenvalues for real and imaginary parts with the starting point as the input
        weight_r = self.linear_r(self.weight.permute(1, 0)).permute(1, 0)
        weight_i = self.linear_i(self.weight.permute(1, 0)).permute(1, 0)

        # apply learned eigenvalues to the real and imaginary parts of the input data
        reweighted_fft_real = data_fft.real * weight_r
        reweighted_fft_imag = data_fft.imag * weight_i

        # transform data back to time domain
        reweighted_fft = torch.complex(reweighted_fft_real, reweighted_fft_imag)
        reweighted_input = torch.fft.ifft(reweighted_fft, dim=1, n=input.size(1)).real
        return reweighted_input
