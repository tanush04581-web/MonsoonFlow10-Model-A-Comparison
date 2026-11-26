"""Phase 4 GPU Curriculum Trainer for MonsoonFlow-10.

Trains candidate architectures strictly on 2005-2018 training data with
progressive rollout curriculum (1 -> 3 -> 5 -> 10 days), physics-aware loss,
validation monitoring on 2019-2020, early stopping, optimizer state saving,
and full reproducibility logging. Never reads or evaluates the 2021 test split.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from monsoon_r1 import contract as C
    from monsoon_r1 import scoring as S
except ImportError:  # pragma: no cover
    _KIT_DIR = Path(__file__).resolve().parent.parent / "resources" / "yug-monsoon-kit"
    if _KIT_DIR.exists() and str(_KIT_DIR) not in sys.path:
        sys.path.insert(0, str(_KIT_DIR))
    from monsoon_r1 import contract as C
    from monsoon_r1 import scoring as S

from models import MonsoonPhysicsLoss, build_model
from models import transforms as T
from training.evaluate_val import evaluate_model_full, evaluate_shocks_and_cpu_latency, predict_validation_rollouts


def _system_ram_gb() -> dict[str, float]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {
        "total_gb": round(m.ullTotalPhys / 1e9, 3),
        "avail_gb": round(m.ullAvailPhys / 1e9, 3),
        "load_pct": float(m.dwMemoryLoad),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _load_split_tensors(
    npz_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load split arrays into normalized CPU float32 tensors + cyclic DOY pairs."""
    with np.load(npz_path) as z:
        data_phys = np.asarray(z["data"], dtype=np.float32)
        doy_int = np.asarray(z["doy"], dtype=np.int32)

    m = mean.reshape(1, C.NC, 1, 1)
    s = np.maximum(std.reshape(1, C.NC, 1, 1), T.STD_FLOOR)
    norm_data = ((data_phys - m) / s).astype(np.float32, copy=False)

    theta = 2.0 * np.pi * doy_int.astype(np.float64) / 365.0
    doy_sc = np.stack([np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32)
    return torch.from_numpy(norm_data), torch.from_numpy(doy_sc)


def _make_window_loader(
    norm_data: torch.Tensor,
    doy_sc: torch.Tensor,
    rollout_steps: int,
    batch_size: int,
    shuffle: bool,
    stride: int = 1,
    seed: int = 2026,
) -> DataLoader:
    """Create lightweight index DataLoader without duplicating sliding windows in RAM."""
    win_len = C.T_IN + rollout_steps
    n_days = norm_data.shape[0]
    starts = torch.arange(0, n_days - win_len + 1, stride, dtype=torch.long)
    ds = TensorDataset(starts)
    gen = torch.Generator()
    gen.manual_seed(seed + rollout_steps)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen if shuffle else None, num_workers=0)


def _gather_batch(
    norm_data: torch.Tensor,
    doy_sc: torch.Tensor,
    starts: torch.Tensor,
    rollout_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather (B, 3, 6, 32, 36), (B, 2), and (B, steps, 6, 32, 36) directly to GPU."""
    # Avoid copying huge tensors to GPU; index tiny slices on CPU and ship batch only
    hist_offsets = torch.arange(C.T_IN, dtype=torch.long)[None, :] + starts[:, None]
    tgt_offsets = torch.arange(C.T_IN, C.T_IN + rollout_steps, dtype=torch.long)[None, :] + starts[:, None]
    hist = norm_data[hist_offsets].to(device, non_blocking=True)
    tgt = norm_data[tgt_offsets].to(device, non_blocking=True)
    t0_doy = doy_sc[starts + (C.T_IN - 1)].to(device, non_blocking=True)
    return hist, t0_doy, tgt


@torch.no_grad()
def _quick_val_metrics(
    model: torch.nn.Module,
    val_npz: Path,
    baselines_cache: dict[str, np.ndarray],
    loss_fn: MonsoonPhysicsLoss,
    val_norm: torch.Tensor,
    val_doy_sc: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate fast full-719-window 10-day validation metrics for model selection."""
    model.eval()
    # 1. Normalized validation 10-step loss
    val_loader = _make_window_loader(val_norm, val_doy_sc, rollout_steps=C.SCORE_STEPS, batch_size=64, shuffle=False)
    loss_sum = 0.0
    n_batches = 0
    for (starts,) in val_loader:
        bx, bd, bt = _gather_batch(val_norm, val_doy_sc, starts, C.SCORE_STEPS, device)
        pred_norm = model.rollout_normalized(bx, bd, steps=C.SCORE_STEPS)
        loss, _ = loss_fn(pred_norm, bt)
        loss_sum += float(loss.item())
        n_batches += 1
    val_loss = loss_sum / max(n_batches, 1)

    # 2. Physical validation metrics matching official scoring.py
    y_pred, _ = predict_validation_rollouts(model, val_npz=val_npz, device=str(device), batch_size=64)
    y_true = baselines_cache["y_true"]
    y_persist = baselines_cache["y_persist"]
    y_clim = baselines_cache["y_clim"]

    valid_ok = all(not S.validity(y_pred[w], C.SCORE_STEPS) for w in range(len(y_pred)))

    ss_list = []
    csi_ss_list = []
    for day in C.LEAD_DAYS:
        k = day - 1
        sk = S.skill_vs_best_ref(y_pred[:, k], y_true[:, k], y_persist[:, k], y_clim[:, k])
        ss_list.append(float(np.nanmean(sk["ss"])))
        csi_m = S.csi(y_pred[:, k, 1], y_true[:, k, 1], thr=S.RAIN_THRESH)
        csi_p = S.csi(y_persist[:, k, 1], y_true[:, k, 1], thr=S.RAIN_THRESH)
        csi_c = S.csi(y_clim[:, k, 1], y_true[:, k, 1], thr=S.RAIN_THRESH)
        csi_ss_list.append(S.csi_skill(csi_m, csi_p, csi_c))

    fcst = S.forecast_points(ss_list, csi_ss_list)
    ratios_d8 = np.asarray([S.variance_ratio(y_pred[w, 7], y_true[w, 7]) for w in range(len(y_pred))])
    anti_blur = S.anti_blur_points(ratios_d8, max_pts=10.0)
    mean_vr = np.nanmean(ratios_d8, axis=0)

    d4_pred_rmse = float(np.mean(S.rmse_lat(y_pred[:, 3], y_true[:, 3])))
    d4_pers_rmse = float(np.mean(S.rmse_lat(y_persist[:, 3], y_true[:, 3])))

    # Selection score combines official forecast skill (45 pts) and anti-blur (10 pts)
    sel_score = float(fcst["points"] + anti_blur) if valid_ok else -100.0
    return {
        "val_loss": float(val_loss),
        "valid_ok": bool(valid_ok),
        "beat_day4_persist": bool(d4_pred_rmse < d4_pers_rmse),
        "mean_ss_148": float(fcst["mean_ss"]),
        "mean_csi_ss_148": float(fcst["mean_csi_skill"]),
        "rmse_pts": float(fcst["rmse_pts"]),
        "csi_pts": float(fcst["csi_pts"]),
        "anti_blur_pts": float(anti_blur),
        "day8_vr_min": float(np.nanmin(mean_vr)),
        "day8_vr_max": float(np.nanmax(mean_vr)),
        "day8_vr_tp": float(mean_vr[1]),
        "selection_score_55": sel_score,
        "ss_by_lead": {f"day_{d}": float(v) for d, v in zip(C.LEAD_DAYS, ss_list)},
        "csi_ss_by_lead": {f"day_{d}": float(v) for d, v in zip(C.LEAD_DAYS, csi_ss_list)},
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    stats: dict[str, Any],
    run_config: dict[str, Any],
    val_metrics: dict[str, Any] | None = None,
    epoch: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "architecture": model.arch_name,
        "model_config": model.model_config(),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "stats": stats,
        "tp_transform": "linear",
        "orography_scale_m": float(model.orography_scale_m),
        "epoch": int(epoch),
        "run_config": run_config,
        "val_metrics": val_metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def train_experiment(
    run_name: str,
    arch: str,
    use_seasonal_anchor: bool = False,
    batch_size: int = 24,
    grad_accum: int = 1,
    stage_epochs: tuple[int, int, int, int] = (4, 4, 4, 10),
    stage_lrs: tuple[float, float, float, float] = (1.5e-3, 9e-4, 6e-4, 3.5e-4),
    stage4_stride: int = 1,
    weight_decay: float = 1e-4,
    w_forecast: float = 1.0,
    w_rain: float = 0.45,
    w_var: float = 0.35,
    w_grad: float = 0.25,
    patience: int = 5,
    seed: int = 2026,
    out_root: Path = Path("checkpoints/runs"),
) -> dict[str, Any]:
    """Run one complete curriculum experiment on RTX 4060 GPU."""
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for Phase 4 training.")

    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path("data/raw/splits/train_stats.json").read_text(encoding="utf-8"))
    mean_np = np.asarray(stats["mean"], dtype=np.float32)
    std_np = np.asarray(stats["std"], dtype=np.float32)
    static_np = np.load("data/raw/splits/static.npy").astype(np.float32)

    train_norm, train_doy_sc = _load_split_tensors(Path("data/raw/splits/train.npz"), mean_np, std_np)
    val_npz = Path("data/raw/splits/val.npz")
    val_norm, val_doy_sc = _load_split_tensors(val_npz, mean_np, std_np)

    from training.baselines import build_validation_rollouts

    val_cache = build_validation_rollouts(val_npz=val_npz, window_stride=1)

    model = build_model(
        arch=arch,
        mean=stats["mean"],
        std=stats["std"],
        static_raw=torch.from_numpy(static_np),
        use_seasonal_anchor=use_seasonal_anchor,
    ).to(device)

    loss_fn = MonsoonPhysicsLoss(
        mean=stats["mean"],
        std=stats["std"],
        w_forecast=w_forecast,
        w_rain=w_rain,
        w_var=w_var,
        w_grad=w_grad,
    ).to(device)

    run_config = {
        "run_name": run_name,
        "arch": arch,
        "use_seasonal_anchor": use_seasonal_anchor,
        "model_config": model.model_config(),
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "stage_steps": [1, 3, 5, 10],
        "stage_epochs": list(stage_epochs),
        "stage_lrs": list(stage_lrs),
        "stage4_stride": stage4_stride,
        "weight_decay": weight_decay,
        "loss_weights": {
            "w_forecast": w_forecast,
            "w_rain": w_rain,
            "w_var": w_var,
            "w_grad": w_grad,
        },
        "patience": patience,
        "seed": seed,
        "precision": "fp32",
        "device_name": str(torch.cuda.get_device_name(0)),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lrs[0], weight_decay=weight_decay)

    curriculum = [
        (1, stage_epochs[0], stage_lrs[0], 1),
        (3, stage_epochs[1], stage_lrs[1], 1),
        (5, stage_epochs[2], stage_lrs[2], 1),
        (10, stage_epochs[3], stage_lrs[3], stage4_stride),
    ]

    history: list[dict[str, Any]] = []
    best_sel_score = -1e9
    best_epoch = -1
    best_val_metrics: dict[str, Any] | None = None
    no_improve_stage4 = 0
    global_epoch = 0

    print(
        f"\n[train] Starting '{run_name}' ({arch}, anchor={use_seasonal_anchor}, "
        f"params={run_config['param_count']}) | free_RAM={_system_ram_gb()['avail_gb']:.2f} GB"
    )

    for stage_idx, (roll_steps, n_ep, lr_start, stride) in enumerate(curriculum, start=1):
        if n_ep <= 0:
            continue
        for pg in optimizer.param_groups:
            pg["lr"] = lr_start
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_ep, eta_min=max(lr_start * 0.15, 1e-5)
        )
        train_loader = _make_window_loader(
            train_norm,
            train_doy_sc,
            rollout_steps=roll_steps,
            batch_size=batch_size,
            shuffle=True,
            stride=stride,
            seed=seed + stage_idx * 100,
        )

        for ep in range(1, n_ep + 1):
            global_epoch += 1
            t_ep = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            model.train()

            loss_accum = 0.0
            term_accum: dict[str, float] = {}
            n_steps_ep = 0
            optimizer.zero_grad(set_to_none=True)

            for b_idx, (starts,) in enumerate(train_loader, start=1):
                bx, bd, bt = _gather_batch(train_norm, train_doy_sc, starts, roll_steps, device)
                pred_norm = model.rollout_normalized(bx, bd, steps=roll_steps)
                loss, terms = loss_fn(pred_norm, bt)
                (loss / grad_accum).backward()

                if b_idx % grad_accum == 0 or b_idx == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                loss_accum += float(loss.item())
                for k, v in terms.items():
                    term_accum[k] = term_accum.get(k, 0.0) + v
                n_steps_ep += 1

            scheduler.step()
            torch.cuda.synchronize()
            ep_time_s = time.perf_counter() - t_ep
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

            # Evaluate 10-day validation after every Stage-4 epoch and at end of Stages 1-3
            do_val = (stage_idx == 4) or (ep == n_ep) or (ep % 2 == 0)
            val_m = (
                _quick_val_metrics(model, val_npz, val_cache, loss_fn, val_norm, val_doy_sc, device)
                if do_val
                else None
            )

            rec = {
                "global_epoch": global_epoch,
                "stage": stage_idx,
                "stage_steps": roll_steps,
                "stage_epoch": ep,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": loss_accum / max(n_steps_ep, 1),
                "train_terms": {k: v / max(n_steps_ep, 1) for k, v in term_accum.items()},
                "epoch_time_s": round(ep_time_s, 2),
                "peak_vram_mb": round(peak_vram_mb, 1),
                "val": val_m,
            }
            history.append(rec)

            if val_m is not None:
                print(
                    f"  S{stage_idx}(k={roll_steps:2d}) E{ep:2d}/{n_ep} | "
                    f"tr_loss={rec['train_loss']:.4f} val_loss={val_m['val_loss']:.4f} | "
                    f"SS={val_m['mean_ss_148']:+.4f} CSI_SS={val_m['mean_csi_ss_148']:+.4f} "
                    f"AB={val_m['anti_blur_pts']:.2f}/10 vr=[{val_m['day8_vr_min']:.2f},{val_m['day8_vr_max']:.2f}] "
                    f"Score55={val_m['selection_score_55']:.2f} | "
                    f"{ep_time_s:.1f}s VRAM={peak_vram_mb:.0f}MB"
                )
                # Track best checkpoint strictly during full 10-step Stage 4 (or as fallback at end of Stage 3)
                if stage_idx == 4 or (best_val_metrics is None):
                    if val_m["selection_score_55"] > best_sel_score:
                        best_sel_score = val_m["selection_score_55"]
                        best_epoch = global_epoch
                        best_val_metrics = val_m
                        no_improve_stage4 = 0
                        _save_checkpoint(
                            run_dir / "best.pt",
                            model=model,
                            stats=stats,
                            run_config=run_config,
                            val_metrics=val_m,
                            epoch=global_epoch,
                        )
                    elif stage_idx == 4:
                        no_improve_stage4 += 1
            else:
                print(
                    f"  S{stage_idx}(k={roll_steps:2d}) E{ep:2d}/{n_ep} | "
                    f"tr_loss={rec['train_loss']:.4f} | {ep_time_s:.1f}s VRAM={peak_vram_mb:.0f}MB"
                )

            _save_checkpoint(
                run_dir / "last_state.pt",
                model=model,
                stats=stats,
                run_config=run_config,
                val_metrics=val_m,
                epoch=global_epoch,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            if stage_idx == 4 and no_improve_stage4 >= patience:
                print(f"  [early_stop] No improvement for {patience} Stage-4 epochs (best epoch={best_epoch}).")
                break

    # Load best checkpoint and run full Phase 5 scientific + shock evaluation
    best_ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    full_eval = evaluate_model_full(model, val_npz=val_npz, device=str(device), run_shocks=True)
    full_eval["best_epoch"] = best_epoch
    full_eval["best_selection_score_55"] = best_sel_score
    full_eval["run_config"] = run_config
    (run_dir / "val_report.json").write_text(json.dumps(full_eval, indent=2), encoding="utf-8")

    s = full_eval["official_scoring_summary"]
    sh = full_eval["shocks_and_cpu"]
    print(
        f"[done] '{run_name}' best_ep={best_epoch} | Score70={s['automated_model_score_out_of_70']:.2f}/70 "
        f"(RMSE={s['rmse_pts_out_of_30']:.2f}/30 [SS={s['mean_rmse_ss_148']:+.4f}], "
        f"CSI={s['csi_pts_out_of_15']:.2f}/15 [CSI_SS={s['mean_csi_ss_148']:+.4f}], "
        f"AB={s['anti_blur_pts_out_of_10']:.2f}/10, "
        f"Caus={s['causality_pts_out_of_15']:.2f}/15) | "
        f"CPU_4c={sh['cpu_4thread_latency_ms']['median']:.1f}ms"
    )
    return full_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MonsoonFlow-10 candidate on GPU.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--arch", type=str, required=True, choices=["mf10_convlstm", "st_rescnn", "ms_resunet"])
    parser.add_argument("--seasonal-anchor", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--stage-epochs", type=int, nargs=4, default=[4, 4, 4, 10])
    parser.add_argument("--stage-lrs", type=float, nargs=4, default=[1.5e-3, 9e-4, 6e-4, 3.5e-4])
    parser.add_argument("--stage4-stride", type=int, default=1)
    parser.add_argument("--w-rain", type=float, default=0.45)
    parser.add_argument("--w-var", type=float, default=0.35)
    parser.add_argument("--w-grad", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    train_experiment(
        run_name=args.run_name,
        arch=args.arch,
        use_seasonal_anchor=args.seasonal_anchor,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        stage_epochs=tuple(args.stage_epochs),
        stage_lrs=tuple(args.stage_lrs),
        stage4_stride=args.stage4_stride,
        w_rain=args.w_rain,
        w_var=args.w_var,
        w_grad=args.w_grad,
        patience=args.patience,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
