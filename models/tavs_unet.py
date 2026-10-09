import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNetModel, timestep_embedding


DEFAULT_MODEL_CHN = 128
DEFAULT_MR_BASE_CH = 32
DEFAULT_TAVS_NUM_DIRS = 8
DEFAULT_CHANNEL_MULT = (1, 1, 2, 3, 4)


class ZeroConv2d(nn.Conv2d):


  def reset_parameters(self):
    nn.init.zeros_(self.weight)
    if self.bias is not None:
      nn.init.zeros_(self.bias)


class ConvSiLU(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
      ),
      nn.SiLU(),
      nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
      nn.SiLU(),
    )

  def forward(self, x):
    return self.block(x)


class MREncoder(nn.Module):


  def __init__(self, in_ch=1, base_ch=DEFAULT_MR_BASE_CH):
    super().__init__()
    self.enc256 = ConvSiLU(in_ch, base_ch, stride=1)
    self.enc128 = ConvSiLU(base_ch, base_ch * 2, stride=2)
    self.enc64 = ConvSiLU(base_ch * 2, base_ch * 4, stride=2)
    self.enc32 = ConvSiLU(base_ch * 4, base_ch * 4, stride=2)
    self.enc16 = ConvSiLU(base_ch * 4, base_ch * 4, stride=2)

  def forward(self, mr):
    m256 = self.enc256(mr)
    m128 = self.enc128(m256)
    m64 = self.enc64(m128)
    m32 = self.enc32(m64)
    m16 = self.enc16(m32)
    return {
      "m256": m256,
      "m128": m128,
      "m64": m64,
      "m32": m32,
      "m16": m16,
    }


class DirectionalVelocityBasis(nn.Module):


  def __init__(self, channels, num_dirs=DEFAULT_TAVS_NUM_DIRS):
    super().__init__()
    self.branches = nn.ModuleList(
      [
        nn.Sequential(
          nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
          ),
          nn.Conv2d(channels, channels, kernel_size=1),
          nn.SiLU(),
        )
        for _ in range(num_dirs)
      ]
    )

  def forward_weighted(self, h, omega):
    steered = None
    for index, branch in enumerate(self.branches):
      candidate = branch(h)
      weight = omega[:, index : index + 1]
      steered = candidate * weight if steered is None else steered + candidate * weight
    return steered


class TAVSBlock(nn.Module):


  def __init__(
      self,
      h_ch,
      mr_ch,
      emb_ch,
      num_dirs=DEFAULT_TAVS_NUM_DIRS,
      inner_ch=None,
  ):
    super().__init__()
    if inner_ch is None:
      inner_ch = min(64, h_ch, mr_ch)

    self.h_proj = nn.Conv2d(h_ch, inner_ch, kernel_size=1)
    self.mr_proj = nn.Conv2d(mr_ch, inner_ch, kernel_size=1)
    self.dir_head = nn.Sequential(
      nn.Conv2d(inner_ch * 2, inner_ch, kernel_size=3, padding=1),
      nn.SiLU(),
      nn.Conv2d(inner_ch, num_dirs, kernel_size=1),
    )
    self.conf_head = nn.Sequential(
      nn.Conv2d(inner_ch * 2, inner_ch, kernel_size=3, padding=1),
      nn.SiLU(),
      nn.Conv2d(inner_ch, 1, kernel_size=1),
    )

    self.basis_ch = min(32, max(16, h_ch // 4))
    self.basis_in = nn.Sequential(
      nn.Conv2d(h_ch, self.basis_ch, kernel_size=1),
      nn.SiLU(),
    )
    self.dir_basis = DirectionalVelocityBasis(
      self.basis_ch, num_dirs=num_dirs
    )
    self.time_gate = nn.Sequential(
      nn.SiLU(),
      nn.Linear(emb_ch, self.basis_ch),
    )
    self.out = ZeroConv2d(self.basis_ch, h_ch, kernel_size=1)
    self.record_visuals = False
    self.last_visuals = {}

  def forward(self, h, mr, emb):
    batch, _channels, height, width = h.shape
    if mr.shape[-2:] != (height, width):
      mr = F.interpolate(
        mr, size=(height, width), mode="bilinear", align_corners=False
      )
    mr = mr.to(dtype=h.dtype)

    h_proj = self.h_proj(h)
    mr_proj = self.mr_proj(mr)
    joint = torch.cat([h_proj, mr_proj], dim=1)
    omega = torch.softmax(self.dir_head(joint), dim=1)
    rho = torch.sigmoid(self.conf_head(joint))

    basis_feature = self.basis_in(h)
    steered = rho * self.dir_basis.forward_weighted(basis_feature, omega)
    gate = torch.sigmoid(self.time_gate(emb)).view(
      batch, self.basis_ch, 1, 1
    )
    gate = gate.to(dtype=h.dtype)
    delta = self.out(gate * steered)

    if self.record_visuals:
      self.last_visuals = {
        "rho": rho.detach().float().cpu(),
        "omega": omega.detach().float().cpu(),
        "delta": delta.detach().float().cpu(),
        "gate": gate.detach().float().cpu(),
      }
    return h + delta


class TAVSUNetModel(UNetModel):


  def __init__(
      self,
      *args,
      image_size=256,
      mr_base_ch=DEFAULT_MR_BASE_CH,
      tavs_num_dirs=DEFAULT_TAVS_NUM_DIRS,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    if not self.time_emb:
      raise ValueError("TAVSUNetModel requires time_emb=True")

    self.image_size = image_size
    self.mr_encoder = MREncoder(in_ch=1, base_ch=mr_base_ch)
    emb_ch = self.model_chn * 4

    self.tavs_mid = TAVSBlock(
      h_ch=self.model_chn * 4,
      mr_ch=mr_base_ch * 4,
      emb_ch=emb_ch,
      num_dirs=tavs_num_dirs,
    )
    self.tavs_64 = TAVSBlock(
      h_ch=self.model_chn * 2,
      mr_ch=mr_base_ch * 4,
      emb_ch=emb_ch,
      num_dirs=tavs_num_dirs,
    )
    self.tavs_128 = TAVSBlock(
      h_ch=self.model_chn,
      mr_ch=mr_base_ch * 2,
      emb_ch=emb_ch,
      num_dirs=tavs_num_dirs,
    )
    self.tavs_256 = TAVSBlock(
      h_ch=self.model_chn,
      mr_ch=mr_base_ch,
      emb_ch=emb_ch,
      num_dirs=tavs_num_dirs,
    )

  def named_tavs_blocks(self):
    return (
      ("bottleneck", self.tavs_mid),
      ("64", self.tavs_64),
      ("128", self.tavs_128),
      ("256", self.tavs_256),
    )

  def set_tavs_visualization(self, enabled=True):
    for _scale, block in self.named_tavs_blocks():
      block.record_visuals = enabled
      if not enabled:
        block.last_visuals = {}

  def get_tavs_visuals(self):
    return {
      scale: block.last_visuals
      for scale, block in self.named_tavs_blocks()
      if block.last_visuals
    }

  def prepare_condition(self, mr):
    return {"mr_feats": self.mr_encoder(mr)}

  @staticmethod
  def _prepare_timesteps(timesteps, batch, device):
    if timesteps is None:
      raise ValueError("timesteps are required")
    while timesteps.dim() > 1:
      timesteps = timesteps[:, 0]
    if timesteps.dim() == 0:
      timesteps = timesteps.repeat(batch)
    elif timesteps.shape[0] == 1 and batch > 1:
      timesteps = timesteps.expand(batch)
    elif timesteps.shape[0] != batch:
      raise ValueError(
        f"Expected one timestep or {batch} timesteps, "
        f"but received {timesteps.shape[0]}"
      )
    return timesteps.to(device=device)

  def forward(self, x, timesteps=None, y=None, condition=None):
    if y is None:
      raise ValueError("TAVS requires the source MR condition y")
    timesteps = self._prepare_timesteps(timesteps, x.shape[0], x.device)
    emb = self.time_embed(timestep_embedding(timesteps, self.model_chn))
    condition = self.prepare_condition(y) if condition is None else condition
    mr_feats = condition["mr_feats"]

    hs = []
    h = torch.cat([x, y], dim=1)
    for block in self.input_blocks:
      h = block(h, emb)
      hs.append(h)

    h = self.middle_block(h, emb)
    h = self.tavs_mid(h, mr_feats["m16"], emb)

    applied_64 = False
    applied_128 = False
    applied_256 = False
    for block in self.output_blocks:
      h = torch.cat([h, hs.pop()], dim=1)
      h = block(h, emb)
      resolution = h.shape[-1]
      if (not applied_64) and resolution == self.image_size // 4:
        h = self.tavs_64(h, mr_feats["m64"], emb)
        applied_64 = True
      elif (not applied_128) and resolution == self.image_size // 2:
        h = self.tavs_128(h, mr_feats["m128"], emb)
        applied_128 = True
      elif (not applied_256) and resolution == self.image_size:
        h = self.tavs_256(h, mr_feats["m256"], emb)
        applied_256 = True

    return self.out(h)


class MRCTRectifiedFlowUNet(nn.Module):


  def __init__(
      self,
      image_size=256,
      model_chn=DEFAULT_MODEL_CHN,
      num_res_blocks=2,
      attention_resolutions=(0,),
      channel_mult=DEFAULT_CHANNEL_MULT,
      dropout=0.0,
      use_scale_shift_norm=True,
      resblock_updown=True,
      mr_base_ch=DEFAULT_MR_BASE_CH,
      tavs_num_dirs=DEFAULT_TAVS_NUM_DIRS,
  ):
    super().__init__()
    common = dict(
      in_chn=2,
      out_chn=2,
      model_chn=model_chn,
      num_res_blocks=num_res_blocks,
      attention_resolutions=attention_resolutions,
      channel_mult=channel_mult,
      dropout=dropout,
      use_scale_shift_norm=use_scale_shift_norm,
      resblock_updown=resblock_updown,
      time_emb=True,
      dims=2,
    )
    self.unet = TAVSUNetModel(
      **common,
      image_size=image_size,
      mr_base_ch=mr_base_ch,
      tavs_num_dirs=tavs_num_dirs,
    )
    self.to_velocity = nn.Conv2d(2, 1, kernel_size=1)

  @staticmethod
  def _prepare_t(t, batch):
    while t.ndim > 1:
      t = t[:, 0]
    if t.ndim == 0:
      t = t.expand(batch)
    elif t.shape[0] == 1 and batch > 1:
      t = t.expand(batch)
    elif t.shape[0] != batch:
      raise ValueError(
        f"Expected one timestep or {batch} timesteps, but received {t.shape[0]}"
      )
    return t

  def prepare_condition(self, mr):
    return self.unet.prepare_condition(mr)

  def set_tavs_visualization(self, enabled=True):
    self.unet.set_tavs_visualization(enabled)

  def get_tavs_visuals(self):
    return self.unet.get_tavs_visuals()

  def forward(self, t, x_t, mr, condition=None):
    if x_t.shape[1] != 1:
      raise ValueError(f"Expected one flow-state channel, got {x_t.shape[1]}")
    t = self._prepare_t(t, x_t.shape[0])
    features = self.unet(
      x=x_t,
      timesteps=t,
      y=mr,
      condition=condition,
    )
    return self.to_velocity(features)


def build_velocity_model(
    model_chn=DEFAULT_MODEL_CHN,
    image_size=256,
    mr_base_ch=DEFAULT_MR_BASE_CH,
    tavs_num_dirs=DEFAULT_TAVS_NUM_DIRS,
):

  if model_chn <= 0 or model_chn % 32 != 0:
    raise ValueError("model_chn must be a positive multiple of 32")
  if image_size <= 0 or image_size % 16 != 0:
    raise ValueError("image_size must be a positive multiple of 16")
  if mr_base_ch <= 0:
    raise ValueError("mr_base_ch must be positive")
  if tavs_num_dirs < 1:
    raise ValueError("tavs_num_dirs must be at least 1")
  return MRCTRectifiedFlowUNet(
    image_size=image_size,
    model_chn=model_chn,
    mr_base_ch=mr_base_ch,
    tavs_num_dirs=tavs_num_dirs,
  )
