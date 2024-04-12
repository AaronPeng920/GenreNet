import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import math
import torchvision.models as models

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class GenreNetBlock(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)

class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels=160, kernel_size=251, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
        band_pass = band_pass / (2*band[:,None])
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)

class SincNet(nn.Module):
    def __init__(self, sinc_in_channels=1, sinc_out_channels=160, sinc_kernel_size=[251, 501, 1001], sinc_sample_rate=16000, num_classes=10):
        super().__init__()
        
        self.sinc_conv1 = nn.Sequential(
            SincConv(out_channels=sinc_out_channels, kernel_size=sinc_kernel_size[0], sample_rate=sinc_sample_rate, in_channels=sinc_in_channels),
            nn.BatchNorm1d(sinc_out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024)
        )
        
        self.sinc_conv2 = nn.Sequential(
            SincConv(out_channels=sinc_out_channels, kernel_size=sinc_kernel_size[1], sample_rate=sinc_sample_rate, in_channels=sinc_in_channels),
            nn.BatchNorm1d(sinc_out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024)
        )
        
        self.sinc_conv3 = nn.Sequential(
            SincConv(out_channels=sinc_out_channels, kernel_size=sinc_kernel_size[2], sample_rate=sinc_sample_rate, in_channels=sinc_in_channels),
            nn.BatchNorm1d(sinc_out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024)
        )
        
        self.cls_head = models.resnet18(pretrained=True)
        self.cls_head.fc = nn.Linear(512, num_classes, bias=True)
        
    def forward(self, waveform):
        out1 = self.sinc_conv1(waveform)
        out2 = self.sinc_conv2(waveform)
        out3 = self.sinc_conv3(waveform)
        out = torch.stack([out1, out2, out3], dim=1)
        out = self.cls_head(out)
        return out
         
        
class GenreNetModule(nn.Module):
    def __init__(
        self, 
        t_seq_len, 
        f_seq_len, 
        t_patch_size, 
        f_patch_size, 
        num_classes, 
        dim, 
        model_dim, 
        depth, 
        heads, 
        mlp_dim, 
        dim_head=64, 
        dropout=0., 
        emb_dropout=0.,
        sinc_in_channels=1,
        sinc_out_channels=160,
        sinc_kernel_size=[251, 501, 1001]
    ):
        super().__init__()
        
        self.t_genreformer = GenreNetBlock(
            seq_len = t_seq_len,
            patch_size = t_patch_size,
            num_classes = model_dim,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout,
            emb_dropout = emb_dropout,
            channels=f_seq_len
        )
        
        self.f_genreformer = GenreNetBlock(
            seq_len = f_seq_len,
            patch_size = f_patch_size,
            num_classes = model_dim,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout,
            emb_dropout = emb_dropout,
            channels = t_seq_len
        )
        
        self.sincnet = SincNet(
            sinc_in_channels=sinc_in_channels, sinc_out_channels=sinc_out_channels, sinc_kernel_size=sinc_kernel_size, 
            sinc_sample_rate=16000, num_classes=model_dim
        )
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(1 * model_dim),
            nn.Linear(1 * model_dim, num_classes)
        )
    
    def forward(self, waveform, melspect):
        waveform = waveform.unsqueeze(1)
        out1 = self.t_genreformer(melspect)
        out2 = self.f_genreformer(melspect.transpose(-1, -2))
        out3 = self.sincnet(waveform)
        out_f = torch.concat([out1, out2, out3], dim=-1)
        pred = self.cls_head(out_f)
        return pred
        