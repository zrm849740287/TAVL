from tqdm import tqdm

import torch
import torch.nn as nn


class RectifiedFlow:
  def __init__(self, loss=nn.MSELoss()):
    self.loss = loss

  @staticmethod
  def euler(x_t, velocity, time_delta):
    return x_t + velocity * time_delta

  def create_flow(self, x_1, t, x_0=None):
    if x_0 is None:
      x_0 = torch.randn_like(x_1)
    t_view = t.view(-1, *([1] * (x_1.ndim - 1)))
    x_t = t_view * x_1 + (1.0 - t_view) * x_0
    return x_t, x_0

  @staticmethod
  def target_velocity(x_1, x_0):
    return x_1 - x_0

  def calc_loss(self, velocity, x_1, x_0, t=None):
    del t
    return self.loss(self.target_velocity(x_1, x_0), velocity)


def exponential_curve(x, k):
  k = torch.tensor(k, dtype=torch.float32, device=x.device)
  if torch.abs(k) < 1e-6:
    return x
  return (torch.exp(k * x) - 1.0) / (torch.exp(k) - 1.0)


def sample_rectified_flow(model, data_in, step, fixed=False, seed=None, k=0):
  flow = RectifiedFlow()
  device = data_in.device
  if fixed:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed if seed is not None else 42)
    x_t = torch.randn(
      data_in.shape,
      generator=generator,
      device=device,
      dtype=data_in.dtype,
    )
  else:
    x_t = torch.randn_like(data_in, device=device)

  condition = data_in.to(device)
  cached_condition = None
  if hasattr(model, "prepare_condition"):
    cached_condition = model.prepare_condition(condition)

  times = exponential_curve(
    torch.linspace(0.0, 1.0, step + 1, device=device), k
  ).view(-1, 1)
  for index, time in tqdm(
      enumerate(times),
      desc="Euler Step",
      leave=False,
      total=step,
      disable=True,
  ):
    if time.item() >= 1.0:
      break
    velocity = model(
      time,
      x_t,
      condition,
      condition=cached_condition,
    )
    time_delta = times[index + 1] - times[index]
    x_t = flow.euler(x_t, velocity, time_delta)
  return x_t
