import torch
import torch.nn as nn


def get_time_embedding(time_steps, t_emb_dim):
    """
    Convert time steps into sinusoidal embeddings.
    :param time_steps: 1D tensor (batch size).
    :param temb_dim: Embedding dimension.
    :return: BxD tensor of time step embeddings.
    """

    assert t_emb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb



class DownBlock(nn.Module):
    """
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample using 2x2 average pooling
    same as hugging face implementation
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample

        # first resnet block
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels,
                              out_channels,
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
                ) for i in range(num_layers)
            ]
        )

        # time embedding
        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ) for _ in range(num_layers)
            ]
        )

        # secnond resnet block
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8,out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
                ) for _ in range(num_layers)
            ]
        )

        # Attention Block
        self.attention_norms = nn.ModuleList(
            [ nn.GroupNorm(8, out_channels) for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )

        # Resdual Connection (input to last conv layer)
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        # Downsample input channel to output channel
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self,x, t_emb):
        out = x
        for i in range(self.num_layers):
            # Resnetblock of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            # Attention block of Unet
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2) # ensures channel features are the last dimension
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn # a residual connection
        out = self.down_sample_conv(out)
        return out
class MidBlock(nn.Module):
    """
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        # first resnet block
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels,
                              out_channels,
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
                ) for i in range(num_layers+1)
            ]
        )

        # time embedding
        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ) for _ in range(num_layers+1)
            ]
        )

        # secnond resnet block
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8,out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
                ) for _ in range(num_layers+1)
            ]
        )
        # Attention Block
        self.attention_norms = nn.ModuleList(
            [ nn.GroupNorm(8, out_channels) for _ in range(num_layers+1)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers+1)]
        )

        # Resdual Connection (input to last conv layer)
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )

    def forward(self,x, t_emb):
        out = x
        
        # Resnetblock 
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            # Attention block 
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2) # ensures channel features are the last dimension
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn # a residual connection
            # Resnetblock 
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)
        return out
    
class UpBlock(nn.Module):
    """
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. up using 2x2 average pooling
    same as hugging face implementation
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample

        # first resnet block
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels,
                              out_channels,
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
                ) for i in range(num_layers)
            ]
        )

        # time embedding
        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ) for _ in range(num_layers)
            ]
        )

        # secnond resnet block
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8,out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
                ) for _ in range(num_layers)
            ]
        )

        # Attention Block
        self.attention_norms = nn.ModuleList(
            [ nn.GroupNorm(8, out_channels) for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )

        # Resdual Connection (input to last conv layer)
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        # Upsample input channel to output channel
        # self.up_sample_conv = nn.Conv2d(out_channels, out_channels,4, 2, 1) if self.up_sample else nn.Identity()
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1) if self.up_sample else nn.Identity()

    def forward(self,x, out_down,t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim = 1)
        out = x
        for i in range(self.num_layers):
            # Resnetblock of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            # Attention block of Unet
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2) # ensures channel features are the last dimension
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn # a residual connection
        #out = self.up_sample_conv(out)
        return out
    

class Unet(nn.Module):
    """
    Unet model comprising ~ Hugging Face
    """
    def __init__(self, model_config):
        super().__init__()

        in_channels = model_config["in_channels"]
        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]

        self.t_emb_dim = model_config["t_emb_dim"]
        self.down_sample = model_config["down_sample"]   
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]
        self.conv_out_channels = model_config["conv_out_channels"]

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(in_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                                        down_sample=self.down_sample[i], num_layers=self.num_down_layers))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                                      num_layers=self.num_mid_layers))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, up_sample=self.down_sample[i], num_layers=self.num_up_layers))
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, in_channels, kernel_size=3, padding=1)
    
    def forward(self,x,t):
        out = self.conv_in(x)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        down_outs = []
    
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
            
        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out
        