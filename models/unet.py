from abc import abstractmethod

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_class(dims, deconv=False, lazy=True):
  if dims != 2:
    raise ValueError(f"Only 2D synthesis is supported, got dims={dims}")
  if lazy:
    return nn.LazyConv2d if not deconv else nn.LazyConvTranspose2d
  return nn.Conv2d if not deconv else nn.ConvTranspose2d


def avg_pool_nd(dims, *args, **kwargs):
  if dims != 2:
    raise ValueError(f"Only 2D synthesis is supported, got dims={dims}")
  return nn.AvgPool2d(*args, **kwargs)


def timestep_embedding(timesteps, dim, max_period=10000):

  half = dim // 2
  freqs = torch.exp(
      -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
  ).to(device=timesteps.device)
  args = timesteps[:, None].float() * freqs[None]
  embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
  if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
  return embedding


def zero_module(module):
  for p in module.parameters():
    p.detach().zero_()
  return module


class Downsample(nn.Module):

  def __init__(self, dims=2, use_conv=False, c=0):
    super().__init__()
    stride = 2
    if use_conv:
      assert c != 0, "c must be specified when use_conv is True"
      self.model = conv_class(dims)(c, kernel_size=3, stride=stride, padding=1)
    else:
      self.model = avg_pool_nd(dims, kernel_size=stride, stride=stride)

  def forward(self, x: torch.Tensor):
    return self.model(x)


class Upsample(nn.Module):

  def __init__(self, dims=2, use_conv=True, c=0):
    super().__init__()
    self.use_conv = use_conv
    self.dims = dims
    if self.use_conv:
      assert c != 0, "c must be specified when use_conv is True"
      self.op = conv_class(dims)(c, kernel_size=3, padding=1)

  def forward(self, x: torch.Tensor):
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    if self.use_conv:
      x = self.op(x)
    return x


class TimestepBlock(nn.Module):

  @abstractmethod
  def forward(self, x, emb=None) -> torch.Tensor:
    pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

  def forward(self, x, emb=None):
    for layer in self:
      if isinstance(layer, TimestepBlock):
        x = layer(x, emb)
      else:
        x = layer(x)
    return x


class ResBlock(TimestepBlock):
  def __init__(self, out_chn, dropout, chn, dims=2, updown=None, emb_dim=0,
               use_scale_shift_norm=False):
    super().__init__()
    self.in_layers = nn.Sequential(
      nn.GroupNorm(32, chn),
      nn.SiLU(),
      zero_module(
        conv_class(dims, lazy=False)(chn, out_chn, kernel_size=3, padding=1)
      )
    )
    self.out_layers = nn.Sequential(
      nn.GroupNorm(32, out_chn),
      nn.SiLU(),
      nn.Dropout(p=dropout),
      zero_module(
        conv_class(dims, lazy=False)(out_chn, out_chn, kernel_size=3, padding=1)
      )
    )
    self.updown = updown is not None
    if updown == 'up':
      self.h_upd = Upsample(dims, use_conv=False)
      self.x_udp = Upsample(dims, use_conv=False)
    elif updown == 'down':
      self.h_upd = Downsample(dims, use_conv=False)
      self.x_udp = Downsample(dims, use_conv=False)
    else:
      self.h_upd = self.x_udp = nn.Identity()

    if chn == out_chn:
      self.res = nn.Identity()
    else:
      self.res = conv_class(dims, lazy=False)(chn, out_chn, kernel_size=1)

    self.emb_dim = emb_dim
    if emb_dim != 0:
      self.use_scale_shift_norm = use_scale_shift_norm
      self.emb_kayers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_dim, 2 * out_chn if use_scale_shift_norm else out_chn),
      )

  def forward(self, x, emb=None):
    if self.updown:
      in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
      h = in_rest(x)
      h = self.h_upd(h)
      x = self.x_udp(x)
      h = in_conv(h)
    else:
      h = self.in_layers(x)

    if self.emb_dim != 0:
      assert emb is not None
      emb_out = self.emb_kayers(emb).type(h.dtype)
      while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]
      if self.use_scale_shift_norm:
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
      else:
        h = h + emb_out
        h = self.out_layers(h)
    else:
      h = self.out_layers(h)

    return self.res(x) + h


class QKVAttention(nn.Module):
  def __init__(self, n_heads):
    super().__init__()
    self.n_heads = n_heads

  def forward(self, qkv):
    bs, width, length = qkv.shape
    assert width % (3 * self.n_heads) == 0
    ch = width // (3 * self.n_heads)
    q, k, v = qkv.chunk(3, dim=1)
    scale = 1 / math.sqrt(math.sqrt(ch))
    weight = torch.einsum(
      "bct, bcs->bts",
      (q * scale).view(bs * self.n_heads, ch, length),
      (k * scale).view(bs * self.n_heads, ch, length)
    )
    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    a = torch.einsum(
      "bts, bcs->bct",
      weight,
      v.reshape(bs * self.n_heads, ch, length)
    )
    return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
  def __init__(self, chn, num_heads=1, num_head_channels=-1):
    super().__init__()
    if num_head_channels == -1:
      self.num_heads = num_heads
    else:
      assert chn % num_head_channels == 0
      self.num_heads = chn// num_head_channels
    self.norm = nn.GroupNorm(32, chn)
    self.qkv = conv_class(dims=1, lazy=False)(chn, chn*3, kernel_size=1)
    self.attention = QKVAttention(self.num_heads)
    self.proj_out = zero_module(conv_class(dims=1, lazy=False)(chn, chn, kernel_size=1))

  def forward(self, x):
    b, c, *spatial = x.shape
    x = x.reshape(b, c, -1)
    qkv = self.qkv(self.norm(x))
    h = self.attention(qkv)
    h = self.proj_out(h)
    return (x + h).reshape(b, c, *spatial)

class UNetModel(nn.Module):

  def __init__(
      self, in_chn, out_chn,
      model_chn, num_res_blocks, attention_resolutions,
      dropout=0., channel_mult=(1, 2, 4, 8), use_scale_shift_norm=False,
      conv_resample=True, dims=2, num_heads=1, num_head_channels=-1,
      time_emb=False, resblock_updown=False
  ):
    super().__init__()
    self.model_chn = model_chn
    self.time_emb = time_emb
    if time_emb:
      time_emb_dim = model_chn * 4
      self.time_embed = nn.Sequential(
        nn.Linear(model_chn, time_emb_dim),
        nn.SiLU(),
        nn.Linear(time_emb_dim, time_emb_dim),
      )

    ch = input_ch = int(channel_mult[0] * model_chn)
    self.input_blocks = nn.ModuleList([
      TimestepEmbedSequential(conv_class(dims, lazy=False)(in_chn, ch, kernel_size=3, padding=1))
    ])
    self._feature_size = ch
    input_block_chans = [ch]
    ds = 1
    for level, mult in enumerate(channel_mult):
      for _ in range(num_res_blocks):
        layers = []
        layers.append(
          ResBlock(
            out_chn=int(mult * model_chn), dropout=dropout, emb_dim=time_emb_dim if time_emb else 0,
            use_scale_shift_norm=use_scale_shift_norm, chn=ch, dims=dims
          )
        )
        ch = int(mult * model_chn)
        if ds in attention_resolutions:
          layers.append(
            AttentionBlock(
              ch, num_heads=num_heads, num_head_channels=num_head_channels,
            )
          )
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        self._feature_size += ch
        input_block_chans.append(ch)
      if level != len(channel_mult) - 1:
        out_ch = ch
        self.input_blocks.append(
          TimestepEmbedSequential(
            ResBlock(
              out_ch, dropout=dropout, chn=ch, dims=dims, emb_dim=time_emb_dim if time_emb else 0,
              updown='down', use_scale_shift_norm=use_scale_shift_norm
            )
            if resblock_updown
            else Downsample(dims=dims, use_conv=conv_resample, c=out_ch)
          )
        )
        ch = out_ch
        input_block_chans.append(ch)
        ds *= 2
        self._feature_size += ch

    self.middle_block = TimestepEmbedSequential(
      ResBlock(
        ch, dropout=dropout, chn=ch, dims=dims, emb_dim=time_emb_dim if time_emb else 0,
        use_scale_shift_norm=use_scale_shift_norm
      ),
      ResBlock(
        ch, dropout=dropout, chn=ch, dims=dims, emb_dim=time_emb_dim if time_emb else 0,
        use_scale_shift_norm=use_scale_shift_norm
      )
    )
    self._feature_size += ch

    self.output_blocks = nn.ModuleList([])
    for level, mult in list(enumerate(channel_mult))[::-1]:
      for i in range(num_res_blocks + 1):
        inch = input_block_chans.pop()
        if len(input_block_chans):
          outch = input_block_chans.pop()
        else:
          outch = inch
        layers = []
        layers.append(
          ResBlock(
            chn=inch * 2, dropout=dropout, out_chn=outch, dims=dims, emb_dim=time_emb_dim if time_emb else 0,
            use_scale_shift_norm=use_scale_shift_norm
          )
        )
        if ds in attention_resolutions:
          layers.append(
            AttentionBlock(
              outch, num_heads=num_heads, num_head_channels=num_head_channels
            )
          )
        if level and i == num_res_blocks:
          layers.append(
            ResBlock(
              chn=outch, out_chn=outch, dropout=dropout, dims=dims, emb_dim=time_emb_dim if time_emb else 0,
              use_scale_shift_norm=use_scale_shift_norm, updown='up'
            )
            if resblock_updown
            else Upsample(dims=dims, use_conv=conv_resample, c=outch)
          )
          ds //= 2
        self.output_blocks.append(TimestepEmbedSequential(*layers))
        self._feature_size += outch
        input_block_chans.append(outch)

    self.out = nn.Sequential(
      nn.GroupNorm(32, outch),
      nn.SiLU(),
      zero_module(conv_class(dims=dims, lazy=False)(input_ch, out_chn, kernel_size=3, padding=1))
    )

  def forward(self, x, timesteps=None, y=None):
    hs = []
    if self.time_emb:
      assert timesteps is not None
      emb = self.time_embed(timestep_embedding(timesteps, self.model_chn))
    else:
      emb = None

    if y is not None:
      h = torch.concat([x, y], dim=1)
    else:
      h = x
    for block in self.input_blocks:
      h = block(h, emb)
      hs.append(h)
    h = self.middle_block(h, emb)
    for block in self.output_blocks:
      h = torch.cat([h, hs.pop()], dim=1)
      h = block(h, emb)
    return self.out(h)


class UNetFP16(UNetModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fp16 = False

  def forward(self, x, timesteps=None, y=None):
    with torch.autocast('cuda', enabled=self.fp16):
      h = super().forward(x, timesteps, y)
    return h

if __name__ == '__main__':
  from torchinfo import summary
  with torch.autocast('cuda'):
    model = UNetModel(1, 1, model_chn=32, num_res_blocks=2, attention_resolutions=(0,), channel_mult=(1, 1, 2, 3, 4),
                      use_scale_shift_norm=True, resblock_updown=True, time_emb=False).cuda()
    summary(model, [(2, 1, 256, 256), (2,)])
