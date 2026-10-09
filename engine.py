import random
import numpy as np
import torch
from tqdm import tqdm

SC_RANDOM_SEED_OFFSET = 104729

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def capture_rng_state(sc_generator):
  state = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch_cpu": torch.get_rng_state(),
    "self_correction": sc_generator.get_state(),
  }
  if torch.cuda.is_available():
    state["torch_cuda"] = torch.cuda.get_rng_state_all()
  return state


def restore_rng_state(state, sc_generator):
  if not state:
    return
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch_cpu"].cpu())
  if torch.cuda.is_available() and "torch_cuda" in state:
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
  sc_generator.set_state(state["self_correction"].cpu())


def validate_arguments(args):
  if args.model_chn <= 0 or args.model_chn % 32 != 0:
    raise ValueError("--model_chn must be a positive multiple of 32")
  if args.img_size <= 0 or args.img_size % 16 != 0:
    raise ValueError("--img_size must be a positive multiple of 16")
  if args.mr_base_ch <= 0:
    raise ValueError("--mr_base_ch must be positive")
  if args.tavs_num_dirs < 1:
    raise ValueError("--tavs_num_dirs must be at least 1")
  if args.sc_loss_weight < 0.0:
    raise ValueError("--sc_loss_weight must be nonnegative")
  if args.sc_warmup_epochs < 0:
    raise ValueError("--sc_warmup_epochs must be nonnegative")
  if args.sc_ramp_epochs < 1:
    raise ValueError("--sc_ramp_epochs must be at least 1")
  if args.sc_update_interval < 1:
    raise ValueError("--sc_update_interval must be at least 1")
  if not 0.0 < args.sc_min_step_ratio <= args.sc_max_step_ratio < 1.0:
    raise ValueError(
      "Expected 0 < sc_min_step_ratio <= sc_max_step_ratio < 1"
    )
  if not 0.0 < args.sc_max_start_time < 1.0:
    raise ValueError("--sc_max_start_time must be in (0, 1)")
  if not 0.0 < args.sc_correction_fraction <= 1.0:
    raise ValueError("--sc_correction_fraction must be in (0, 1]")


def self_correction_ramp(epoch, warmup_epochs, ramp_epochs):
  if epoch <= warmup_epochs:
    return 0.0
  return min(1.0, (epoch - warmup_epochs) / ramp_epochs)


def self_corrective_rollout(
    model,
    t,
    x_t,
    target,
    condition,
    predicted_velocity,
    target_velocity,
    min_step_ratio,
    max_step_ratio,
    max_start_time,
    correction_fraction,
    sc_generator,
):

  time_values = t.reshape(x_t.shape[0], -1)[:, 0].to(
    device=x_t.device, dtype=x_t.dtype
  )
  eligible_indices = torch.nonzero(
    time_values <= max_start_time, as_tuple=False
  ).flatten()
  if eligible_indices.numel() == 0:
    return None

  start_time = time_values.index_select(0, eligible_indices)
  random_values = torch.rand(
    start_time.shape,
    generator=sc_generator,
    device="cpu",
    dtype=torch.float32,
  ).to(device=start_time.device, dtype=start_time.dtype)
  step_ratio = min_step_ratio + (
    max_step_ratio - min_step_ratio
  ) * random_values
  next_time = start_time + step_ratio * (1.0 - start_time)
  time_delta = (next_time - start_time).view(-1, 1, 1, 1)
  remaining_time = (1.0 - next_time).view(-1, 1, 1, 1)

  selected_x_t = x_t.index_select(0, eligible_indices)
  selected_target = target.index_select(0, eligible_indices)
  selected_condition = condition.index_select(0, eligible_indices)
  selected_velocity = predicted_velocity.detach().index_select(
    0, eligible_indices
  )
  selected_target_velocity = target_velocity.index_select(
    0, eligible_indices
  )

  with torch.no_grad():
    rollout_state = selected_x_t + time_delta * selected_velocity
    ideal_next_state = selected_x_t + time_delta * selected_target_velocity
    rollout_deviation = rollout_state - ideal_next_state
    rollout_offset_rmse = rollout_deviation.square().mean().sqrt()

  correction_velocity = model(
    next_time,
    rollout_state,
    selected_condition,
  )
  correction_residual = (
    remaining_time * (correction_velocity - selected_target_velocity)
    + correction_fraction * rollout_deviation
  )
  horizon_normalizer = remaining_time.square().expand_as(
    correction_residual
  ).sum().clamp_min(1e-8)
  correction_loss = correction_residual.square().sum() / horizon_normalizer

  with torch.no_grad():
    endpoint_projection = (
      rollout_state + remaining_time * correction_velocity.detach()
    )
    desired_endpoint = selected_target + (
      1.0 - correction_fraction
    ) * rollout_deviation
    endpoint_rmse = (
      endpoint_projection - selected_target
    ).square().mean().sqrt()
    damped_endpoint_rmse = (
      endpoint_projection - desired_endpoint
    ).square().mean().sqrt()

  return {
    "loss": correction_loss,
    "rollout_offset_rmse": rollout_offset_rmse,
    "endpoint_rmse": endpoint_rmse,
    "damped_endpoint_rmse": damped_endpoint_rmse,
    "step_ratio": step_ratio.mean().detach(),
    "next_time": next_time.mean().detach(),
    "sample_count": int(eligible_indices.numel()),
  }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    flow,
    device,
    args,
    epoch,
    sc_generator,
):
  model.train()
  current_sc_ramp = self_correction_ramp(
    epoch, args.sc_warmup_epochs, args.sc_ramp_epochs
  )
  totals = {
    "objective": 0.0,
    "main_loss": 0.0,
    "sc_term": 0.0,
    "count": 0,
    "sc_loss": 0.0,
    "sc_active_batches": 0,
    "sc_active_samples": 0,
    "rollout_offset": 0.0,
    "endpoint_rmse": 0.0,
    "damped_endpoint_rmse": 0.0,
  }

  progress = tqdm(loader, unit="batch", desc=f"Epoch {epoch}")
  for batch_index, (condition, target) in enumerate(progress):
    condition = condition.to(device, non_blocking=True)
    x_1 = target.to(device, non_blocking=True)
    t = torch.rand(x_1.shape[0], device=device)
    x_t, x_0 = flow.create_flow(x_1, t)
    target_velocity = flow.target_velocity(x_1, x_0)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
      predicted_velocity = model(t, x_t, condition)
      main_loss = flow.calc_loss(predicted_velocity, x_1, x_0, t)
    if not torch.isfinite(main_loss):
      raise FloatingPointError(
        f"Non-finite main loss at epoch {epoch}, step {batch_index + 1}"
      )

    if scaler.is_enabled():
      scaler.scale(main_loss).backward()
    else:
      main_loss.backward()

    zero = main_loss.detach().new_zeros(())
    sc_loss = zero
    sc_term = zero
    rollout_metrics = None
    should_apply_sc = (
      args.sc_loss_weight > 0.0
      and current_sc_ramp > 0.0
      and (batch_index + 1) % args.sc_update_interval == 0
    )
    if should_apply_sc:
      with torch.autocast(
          "cuda", enabled=args.fp16 and device.type == "cuda"
      ):
        rollout_metrics = self_corrective_rollout(
          model=model,
          t=t,
          x_t=x_t,
          target=x_1,
          condition=condition,
          predicted_velocity=predicted_velocity,
          target_velocity=target_velocity,
          min_step_ratio=args.sc_min_step_ratio,
          max_step_ratio=args.sc_max_step_ratio,
          max_start_time=args.sc_max_start_time,
          correction_fraction=args.sc_correction_fraction,
          sc_generator=sc_generator,
        )
        if rollout_metrics is not None:
          sc_loss = rollout_metrics["loss"]
          sc_term = args.sc_loss_weight * current_sc_ramp * sc_loss
      if rollout_metrics is not None:
        if not torch.isfinite(sc_term):
          raise FloatingPointError(
            f"Non-finite self-correction loss at epoch {epoch}, "
            f"step {batch_index + 1}"
          )
        if scaler.is_enabled():
          scaler.scale(sc_term).backward()
        else:
          sc_term.backward()

    if scaler.is_enabled():
      scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    if scaler.is_enabled():
      scaler.step(optimizer)
      scaler.update()
    else:
      optimizer.step()

    batch_size = x_1.shape[0]
    objective = main_loss.detach() + sc_term.detach()
    totals["objective"] += objective.item() * batch_size
    totals["main_loss"] += main_loss.item() * batch_size
    totals["sc_term"] += sc_term.item() * batch_size
    totals["count"] += batch_size

    if rollout_metrics is not None:
      active_samples = rollout_metrics["sample_count"]
      totals["sc_loss"] += sc_loss.item()
      totals["sc_active_batches"] += 1
      totals["sc_active_samples"] += active_samples
      totals["rollout_offset"] += (
        rollout_metrics["rollout_offset_rmse"].item() * active_samples
      )
      totals["endpoint_rmse"] += (
        rollout_metrics["endpoint_rmse"].item() * active_samples
      )
      totals["damped_endpoint_rmse"] += (
        rollout_metrics["damped_endpoint_rmse"].item() * active_samples
      )

    progress.set_postfix(
      {
        "objective": f"{objective.item():.6f}",
        "main": f"{main_loss.item():.6f}",
        "self_correction": f"{sc_loss.item():.6f}",
        "sc_term": f"{sc_term.item():.6f}",
      }
    )

  count = max(1, totals["count"])
  active_batches = max(1, totals["sc_active_batches"])
  active_samples = max(1, totals["sc_active_samples"])
  return {
    "objective": totals["objective"] / count,
    "main_loss": totals["main_loss"] / count,
    "self_correction_loss": totals["sc_loss"] / active_batches,
    "sc_term": totals["sc_term"] / count,
    "rollout_offset_rmse": totals["rollout_offset"] / active_samples,
    "endpoint_rmse": totals["endpoint_rmse"] / active_samples,
    "damped_endpoint_rmse": totals["damped_endpoint_rmse"] / active_samples,
    "sc_active_batches": totals["sc_active_batches"],
    "sc_active_samples": totals["sc_active_samples"],
    "sc_ramp": current_sc_ramp,
  }
