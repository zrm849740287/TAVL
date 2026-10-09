import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

from models.rectified_flow import RectifiedFlow, sample_rectified_flow
from models.tavs_unet import build_velocity_model
from engine import (
  capture_rng_state,
  restore_rng_state,
  self_corrective_rollout,
  train_one_epoch,
)


class ExactRecoveryModel(nn.Module):
  def __init__(self, target_velocity, velocity_error, start_time, beta):
    super().__init__()
    self.register_buffer("target_velocity", target_velocity)
    self.register_buffer("velocity_error", velocity_error)
    self.start_time = start_time
    self.beta = beta

  def forward(self, t, x, mr):
    del x, mr
    delta = (t - self.start_time).view(-1, 1, 1, 1)
    remaining = (1.0 - t).view(-1, 1, 1, 1)
    deviation = delta * self.velocity_error
    return self.target_velocity - self.beta * deviation / remaining


class LearnableCorrectionModel(nn.Module):
  def __init__(self, target_velocity):
    super().__init__()
    self.register_buffer("target_velocity", target_velocity)
    self.bias = nn.Parameter(torch.full_like(target_velocity, 0.1))

  def forward(self, t, x, mr):
    del t, x, mr
    return self.target_velocity + self.bias


class CachedConditionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.prepare_calls = 0
    self.forward_calls = 0

  def prepare_condition(self, condition):
    self.prepare_calls += 1
    return {"condition": condition}

  def forward(self, t, x, mr, condition=None):
    del t, mr
    self.forward_calls += 1
    if condition is None:
      raise AssertionError("Expected cached condition features")
    return torch.zeros_like(x)


class InnovationTests(unittest.TestCase):
  def test_resume_rng_state_is_exact(self):
    random.seed(21)
    np.random.seed(21)
    torch.manual_seed(21)
    generator = torch.Generator(device="cpu").manual_seed(22)
    state = capture_rng_state(generator)
    expected = (
      random.random(),
      float(np.random.rand()),
      torch.rand(1),
      torch.rand(1, generator=generator),
    )
    restore_rng_state(state, generator)
    actual = (
      random.random(),
      float(np.random.rand()),
      torch.rand(1),
      torch.rand(1, generator=generator),
    )
    self.assertEqual(actual[0], expected[0])
    self.assertEqual(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])
    torch.testing.assert_close(actual[3], expected[3])

  def test_integrated_training_step_with_self_correction(self):
    torch.manual_seed(13)
    model = build_velocity_model(
      model_chn=32,
      image_size=32,
      mr_base_ch=32,
      tavs_num_dirs=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda", enabled=False)
    loader = [
      (
        torch.rand(1, 1, 32, 32),
        torch.rand(1, 1, 32, 32),
      )
    ]
    args = SimpleNamespace(
      sc_warmup_epochs=20,
      sc_ramp_epochs=20,
      fp16=False,
      sc_loss_weight=0.1,
      sc_update_interval=1,
      sc_min_step_ratio=0.1,
      sc_max_step_ratio=0.3,
      sc_max_start_time=0.999999,
      sc_correction_fraction=0.25,
      grad_clip=1.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(17)
    metrics = train_one_epoch(
      model=model,
      loader=loader,
      optimizer=optimizer,
      scaler=scaler,
      flow=RectifiedFlow(),
      device=torch.device("cpu"),
      args=args,
      epoch=21,
      sc_generator=generator,
    )
    self.assertEqual(metrics["sc_active_batches"], 1)
    self.assertEqual(metrics["sc_active_samples"], 1)
    self.assertTrue(torch.isfinite(torch.tensor(metrics["objective"])))

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
  def test_default_256_tavs_forward_on_cuda(self):
    device = torch.device("cuda:0")
    model = build_velocity_model(
      model_chn=128,
      image_size=256,
      mr_base_ch=32,
      tavs_num_dirs=8,
    ).to(device)
    model.eval()
    model.set_tavs_visualization(True)
    with torch.no_grad():
      output = model(
        torch.tensor([0.4], device=device),
        torch.randn(1, 1, 256, 256, device=device),
        torch.randn(1, 1, 256, 256, device=device),
      )
    self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
    visuals = model.get_tavs_visuals()
    expected_sizes = {
      "bottleneck": 16,
      "64": 64,
      "128": 128,
      "256": 256,
    }
    for scale, size in expected_sizes.items():
      self.assertEqual(tuple(visuals[scale]["omega"].shape[-2:]), (size, size))

  def test_tavs_uses_four_locations_and_eight_directions(self):
    torch.manual_seed(1)
    model = build_velocity_model(
      model_chn=32,
      image_size=64,
      mr_base_ch=32,
      tavs_num_dirs=8,
    )
    model.set_tavs_visualization(True)
    output = model(
      torch.tensor([0.4]),
      torch.randn(1, 1, 64, 64),
      torch.randn(1, 1, 64, 64),
    )
    self.assertEqual(tuple(output.shape), (1, 1, 64, 64))
    visuals = model.get_tavs_visuals()
    self.assertEqual(set(visuals), {"bottleneck", "64", "128", "256"})
    for values in visuals.values():
      omega = values["omega"]
      self.assertEqual(omega.shape[1], 8)
      torch.testing.assert_close(
        omega.sum(dim=1),
        torch.ones_like(omega[:, 0]),
        atol=1e-5,
        rtol=1e-5,
      )

  def test_self_correction_is_zero_for_exact_damped_recovery(self):
    batch = 2
    start_time = 0.2
    beta = 0.25
    target_velocity = torch.full((batch, 1, 4, 4), 0.6)
    velocity_error = torch.full_like(target_velocity, 0.2)
    predicted_velocity = target_velocity + velocity_error
    x_t = torch.randn_like(target_velocity)
    t = torch.full((batch,), start_time)
    target = x_t + (1.0 - start_time) * target_velocity
    condition = torch.randn_like(target_velocity)
    model = ExactRecoveryModel(
      target_velocity,
      velocity_error,
      start_time=start_time,
      beta=beta,
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    metrics = self_corrective_rollout(
      model=model,
      t=t,
      x_t=x_t,
      target=target,
      condition=condition,
      predicted_velocity=predicted_velocity,
      target_velocity=target_velocity,
      min_step_ratio=0.1,
      max_step_ratio=0.3,
      max_start_time=0.8,
      correction_fraction=beta,
      sc_generator=generator,
    )
    self.assertIsNotNone(metrics)
    self.assertLess(metrics["loss"].item(), 1e-12)
    self.assertLess(metrics["damped_endpoint_rmse"].item(), 1e-6)

  def test_self_correction_stops_rollout_gradient(self):
    batch = 2
    target_velocity = torch.full((batch, 1, 4, 4), 0.5)
    predicted_velocity = (target_velocity + 0.2).clone().requires_grad_(True)
    x_t = torch.randn_like(target_velocity)
    t = torch.full((batch,), 0.3)
    target = x_t + 0.7 * target_velocity
    condition = torch.randn_like(target_velocity)
    model = LearnableCorrectionModel(target_velocity)
    generator = torch.Generator(device="cpu").manual_seed(9)
    metrics = self_corrective_rollout(
      model=model,
      t=t,
      x_t=x_t,
      target=target,
      condition=condition,
      predicted_velocity=predicted_velocity,
      target_velocity=target_velocity,
      min_step_ratio=0.1,
      max_step_ratio=0.3,
      max_start_time=0.8,
      correction_fraction=0.25,
      sc_generator=generator,
    )
    metrics["loss"].backward()
    self.assertIsNone(predicted_velocity.grad)
    self.assertIsNotNone(model.bias.grad)
    self.assertGreater(model.bias.grad.abs().sum().item(), 0.0)

  def test_sampler_caches_mr_encoder(self):
    model = CachedConditionModel()
    condition = torch.zeros(1, 1, 8, 8)
    output = sample_rectified_flow(
      model, condition, step=4, fixed=True, seed=3, k=0.0
    )
    self.assertEqual(tuple(output.shape), tuple(condition.shape))
    self.assertEqual(model.prepare_calls, 1)
    self.assertEqual(model.forward_calls, 4)

  def test_tavs_sampler_supports_batch_two(self):
    model = build_velocity_model(
      model_chn=32,
      image_size=32,
      mr_base_ch=32,
      tavs_num_dirs=8,
    )
    condition = torch.zeros(2, 1, 32, 32)
    output = sample_rectified_flow(
      model, condition, step=2, fixed=True, seed=3, k=0.0
    )
    self.assertEqual(tuple(output.shape), (2, 1, 32, 32))

  def test_fixed_sampler_does_not_reset_global_rng(self):
    model = CachedConditionModel()
    condition = torch.zeros(1, 1, 8, 8)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    sample_rectified_flow(model, condition, step=2, fixed=True, seed=5, k=0.0)
    actual = torch.rand(3)
    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
  unittest.main()
