import argparse
import copy
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
STATE_DIR = OUTPUT_DIR / "state"

MODEL_FACTORY = {
    "vgg_a": VGG_A,
    "vgg_a_bn": VGG_A_BatchNorm,
    "vgg_a_dropout": VGG_A_Dropout,
}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - target_probs) ** self.gamma) * target_log_probs
        return loss.mean()


def ensure_dirs():
    for path in (OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, METRICS_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_random_seeds(seed_value=0, device="cpu"):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(name, parameters, lr, weight_decay=0.0):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, foreach=False)
    raise ValueError(f"Unsupported optimizer: {name}")


def format_subset_tag(value):
    if value is None or int(value) < 0:
        return "all"
    return str(int(value))


def build_run_name(args, model_name, optimizer_name, lr):
    train_tag = format_subset_tag(args.train_subset)
    val_tag = format_subset_tag(args.val_subset)
    hidden_dim = args.hidden_dim if args.hidden_dim and args.hidden_dim > 0 else "auto"
    loss_tag = "ce" if args.loss_name == "cross_entropy" else args.loss_name
    return (
        f"{model_name}_{optimizer_name}_lr{lr:g}"
        f"_train{train_tag}_val{val_tag}_ep{args.epochs}"
        f"_act{args.activation}"
        f"_w{args.width_multiplier:g}_h{hidden_dim}"
        f"_loss{loss_tag}_ls{args.label_smoothing:g}_wd{args.weight_decay:g}"
    )


def create_model(model_name, args):
    model_cls = MODEL_FACTORY[model_name]
    model_kwargs = {
        "activation_name": args.activation,
        "width_multiplier": args.width_multiplier,
        "hidden_dim": args.hidden_dim if args.hidden_dim and args.hidden_dim > 0 else None,
    }
    if model_name == "vgg_a_dropout":
        model_kwargs["dropout_p"] = args.dropout_p
    return model_cls(**model_kwargs)


def create_criterion(args):
    if args.loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.loss_name == "focal":
        return FocalLoss(gamma=args.focal_gamma)
    raise ValueError(f"Unsupported loss: {args.loss_name}")


def save_training_state(state_path, state):
    torch.save(state, state_path)


def load_training_state(state_path, device):
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location=device, weights_only=False)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_summary(run_name, history, model_name, optimizer_name, lr, history_path, figure_path, checkpoint_path, args):
    return {
        "run_name": run_name,
        "model": model_name,
        "optimizer": optimizer_name,
        "lr": lr,
        "activation": args.activation,
        "width_multiplier": args.width_multiplier,
        "hidden_dim": args.hidden_dim if args.hidden_dim and args.hidden_dim > 0 else None,
        "loss_name": args.loss_name,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "best_val_accuracy": max(history["val_accuracy"]),
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_val_loss": history["val_loss"][-1],
        "epochs_completed": len(history["val_accuracy"]),
        "history_path": str(history_path),
        "figure_path": str(figure_path),
        "checkpoint_path": str(checkpoint_path),
    }


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": running_correct / max(total, 1),
    }


def get_probe_batch(data_loader, device):
    x, y = next(iter(data_loader))
    return x.to(device), y.to(device)


def compute_probe_stats(model, criterion, probe_batch, device):
    model.eval()
    probe_x, probe_y = probe_batch
    model.zero_grad(set_to_none=True)
    logits = model(probe_x)
    loss = criterion(logits, probe_y)
    loss.backward()

    grad_norm_sq = 0.0
    flat_grad_parts = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().flatten()
        flat_grad_parts.append(grad)
        grad_norm_sq += grad.pow(2).sum().item()

    flat_grad = torch.cat(flat_grad_parts) if flat_grad_parts else torch.zeros(1, device=device)
    grad_norm = grad_norm_sq ** 0.5
    return loss.item(), grad_norm, flat_grad.cpu().numpy()


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    epochs_n=10,
    probe_batch=None,
    history=None,
    best_state=None,
    best_val_accuracy=-1.0,
    prev_probe_grad=None,
    start_epoch=0,
    state_path=None,
    run_name=None,
):
    model.to(device)
    if history is None:
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "probe_loss": [],
            "probe_grad_norm": [],
            "probe_grad_delta": [],
        }

    for epoch in tqdm(range(start_epoch, epochs_n), unit="epoch"):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        train_metrics = {
            "loss": running_loss / max(total, 1),
            "accuracy": running_correct / max(total, 1),
        }
        val_metrics = evaluate(model, val_loader, criterion, device)
        probe_loss, probe_grad_norm, probe_grad = compute_probe_stats(model, criterion, probe_batch, device)

        if prev_probe_grad is None:
            probe_grad_delta = 0.0
        else:
            probe_grad_delta = float(np.linalg.norm(probe_grad - prev_probe_grad))

        prev_probe_grad = probe_grad

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["probe_loss"].append(probe_loss)
        history["probe_grad_norm"].append(probe_grad_norm)
        history["probe_grad_delta"].append(probe_grad_delta)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch + 1}/{epochs_n} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if state_path is not None:
            save_training_state(
                state_path,
                {
                    "run_name": run_name,
                    "epoch": epoch + 1,
                    "epochs_n": epochs_n,
                    "history": history,
                    "best_state": best_state,
                    "best_val_accuracy": best_val_accuracy,
                    "prev_probe_grad": prev_probe_grad,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )

    return history, best_state


def save_history(history, output_path):
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)


def collect_metric_summaries(metrics_dir):
    summaries = []
    for metrics_path in sorted(metrics_dir.glob("*.json")):
        if metrics_path.name == "experiment_summary.json":
            continue
        with open(metrics_path, "r", encoding="utf-8") as file_obj:
            history = json.load(file_obj)
        if "val_accuracy" not in history or not history["val_accuracy"]:
            continue
        run_name = metrics_path.stem
        summaries.append({
            "run_name": run_name,
            "best_val_accuracy": max(history["val_accuracy"]),
            "final_val_accuracy": history["val_accuracy"][-1],
            "final_val_loss": history["val_loss"][-1],
        })
    return summaries


def plot_training_curves(history, title, output_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_loss_landscape_curves(histories, key="probe_loss"):
    min_len = min(len(history[key]) for history in histories)
    stacked = np.asarray([history[key][:min_len] for history in histories], dtype=float)
    return stacked.min(axis=0), stacked.max(axis=0)


def plot_loss_landscape(result_map, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, histories in result_map.items():
        min_curve, max_curve = build_loss_landscape_curves(histories, key="probe_loss")
        steps = np.arange(1, len(min_curve) + 1)
        ax.plot(steps, min_curve, label=f"{label} min")
        ax.plot(steps, max_curve, linestyle="--", label=f"{label} max")
        ax.fill_between(steps, min_curve, max_curve, alpha=0.18)

    ax.set_title("Loss Landscape Envelope on Probe Batch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Probe Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_diagnostics(histories_by_model, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, history in histories_by_model.items():
        epochs = np.arange(1, len(history["probe_grad_norm"]) + 1)
        axes[0].plot(epochs, history["probe_grad_norm"], label=label)
        axes[1].plot(epochs, history["probe_grad_delta"], label=label)

    axes[0].set_title("Gradient Norm")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Norm")
    axes[0].legend()

    axes[1].set_title("Gradient Difference Between Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L2 Difference")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_single_experiment(args, model_name, optimizer_name, lr):
    device = get_device()
    set_random_seeds(seed_value=args.seed, device=str(device))
    ensure_dirs()

    train_loader = get_cifar_loader(
        root=str(THIS_DIR / "data"),
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.train_subset,
    )
    val_loader = get_cifar_loader(
        root=str(THIS_DIR / "data"),
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.val_subset,
    )

    run_name = build_run_name(args, model_name, optimizer_name, lr)
    history_path = METRICS_DIR / f"{run_name}.json"
    figure_path = FIGURES_DIR / f"{run_name}_curves.png"
    checkpoint_path = MODELS_DIR / f"{run_name}.pt"
    state_path = STATE_DIR / f"{run_name}.resume.pt"

    if history_path.exists() and not args.force_rerun:
        with open(history_path, "r", encoding="utf-8") as file_obj:
            existing_history = json.load(file_obj)
        if len(existing_history.get("val_accuracy", [])) >= args.epochs:
            print(f"skip completed run: {run_name}")
            summary = build_summary(
                run_name,
                existing_history,
                model_name,
                optimizer_name,
                lr,
                history_path,
                figure_path,
                checkpoint_path,
                args,
            )
            return existing_history, summary

    probe_batch = get_probe_batch(val_loader, device)
    model = create_model(model_name, args)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = create_criterion(args)

    history = None
    best_state = None
    best_val_accuracy = -1.0
    prev_probe_grad = None
    start_epoch = 0
    resume_state = None

    if args.resume and state_path.exists():
        resume_state = load_training_state(state_path, device)
        if resume_state is not None:
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
            history = resume_state.get("history")
            best_state = resume_state.get("best_state")
            best_val_accuracy = resume_state.get("best_val_accuracy", -1.0)
            prev_probe_grad = resume_state.get("prev_probe_grad")
            start_epoch = int(resume_state.get("epoch", 0))
            print(f"resume run: {run_name} from epoch {start_epoch + 1}/{args.epochs}")

    history, best_state = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs_n=args.epochs,
        probe_batch=probe_batch,
        history=history,
        best_state=best_state,
        best_val_accuracy=best_val_accuracy,
        prev_probe_grad=prev_probe_grad,
        start_epoch=start_epoch,
        state_path=state_path,
        run_name=run_name,
    )

    save_history(history, history_path)
    plot_training_curves(history, run_name, figure_path)
    if best_state is not None:
        torch.save(best_state, checkpoint_path)
    if state_path.exists():
        state_path.unlink()

    summary = build_summary(
        run_name,
        history,
        model_name,
        optimizer_name,
        lr,
        history_path,
        figure_path,
        checkpoint_path,
        args,
    )
    return history, summary


def run_landscape_suite(args):
    results = {}
    summaries = []
    representative_histories = {}

    for model_name in args.landscape_models:
        model_histories = []
        representative_history = None
        for lr_index, lr in enumerate(args.landscape_lrs):
            history, summary = run_single_experiment(args, model_name, args.landscape_optimizer, lr)
            model_histories.append(history)
            summaries.append(summary)
            if lr_index == 0:
                representative_history = history
        results[model_name] = model_histories
        if representative_history is not None:
            representative_histories[model_name] = representative_history

    plot_loss_landscape(results, FIGURES_DIR / "loss_landscape_comparison.png")
    plot_gradient_diagnostics(representative_histories, FIGURES_DIR / "gradient_diagnostics.png")
    return summaries


def run_optimizer_suite(args):
    summaries = []
    target_models = args.optimizer_models or args.landscape_models
    for model_name in target_models:
        for optimizer_name in args.optimizers:
            history, summary = run_single_experiment(args, model_name, optimizer_name, args.optimizer_lr)
            summaries.append(summary)
    return summaries


def run_ablation_suites(args):
    summaries = []

    width_pairs = [
        ("narrow", 0.5, 256),
        ("baseline", 1.0, 512),
        ("wide", 1.25, 640),
    ]
    activation_choices = ["relu", "leaky_relu", "elu"]
    loss_settings = [
        ("ce_plain", "cross_entropy", 0.0, 0.0),
        ("ce_weight_decay", "cross_entropy", 0.0, 1e-3),
        ("ce_label_smoothing", "cross_entropy", 0.1, 5e-4),
        ("focal", "focal", 0.0, 5e-4),
    ]

    original_width = args.width_multiplier
    original_hidden = args.hidden_dim
    original_activation = args.activation
    original_loss_name = args.loss_name
    original_label_smoothing = args.label_smoothing
    original_weight_decay = args.weight_decay

    for _, width_multiplier, hidden_dim in width_pairs:
        args.width_multiplier = width_multiplier
        args.hidden_dim = hidden_dim
        args.activation = "relu"
        args.loss_name = "cross_entropy"
        args.label_smoothing = 0.0
        args.weight_decay = 5e-4
        _, summary = run_single_experiment(args, "vgg_a", "adam", args.optimizer_lr)
        summaries.append(summary)

    args.width_multiplier = 1.0
    args.hidden_dim = 512
    for activation_name in activation_choices:
        args.activation = activation_name
        args.loss_name = "cross_entropy"
        args.label_smoothing = 0.0
        args.weight_decay = 5e-4
        _, summary = run_single_experiment(args, "vgg_a", "adam", args.optimizer_lr)
        summaries.append(summary)

    args.activation = "relu"
    for _, loss_name, label_smoothing, weight_decay in loss_settings:
        args.loss_name = loss_name
        args.label_smoothing = label_smoothing
        args.weight_decay = weight_decay
        _, summary = run_single_experiment(args, "vgg_a", "adam", args.optimizer_lr)
        summaries.append(summary)

    args.width_multiplier = original_width
    args.hidden_dim = original_hidden
    args.activation = original_activation
    args.loss_name = original_loss_name
    args.label_smoothing = original_label_smoothing
    args.weight_decay = original_weight_decay

    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="VGG-A / BN experiments on CIFAR-10")
    parser.add_argument("--mode", nargs="+", default=["landscape"], choices=["landscape", "optimizers", "ablations"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-name", default="cross_entropy", choices=["cross_entropy", "focal"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--activation", default="relu", choices=["relu", "leaky_relu", "elu"])
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--train-subset", type=int, default=2000)
    parser.add_argument("--val-subset", type=int, default=1000)
    parser.add_argument(
        "--landscape-models",
        nargs="+",
        default=["vgg_a", "vgg_a_bn"],
        choices=sorted(MODEL_FACTORY.keys()),
    )
    parser.add_argument("--landscape-optimizer", default="adam", choices=["sgd", "sgd_momentum", "adam"])
    parser.add_argument("--landscape-lrs", nargs="+", type=float, default=[1e-3, 2e-3, 5e-4, 1e-4])
    parser.add_argument("--optimizer-models", nargs="+", choices=sorted(MODEL_FACTORY.keys()))
    parser.add_argument("--optimizers", nargs="+", default=["sgd", "sgd_momentum", "adam"], choices=["sgd", "sgd_momentum", "adam"])
    parser.add_argument("--optimizer-lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="resume interrupted runs from per-epoch state files")
    parser.add_argument("--force-rerun", action="store_true", help="rerun experiments even if finished metrics already exist")
    return parser.parse_args()


def main():
    args = parse_args()
    if "landscape" in args.mode:
        run_landscape_suite(args)
    if "optimizers" in args.mode:
        run_optimizer_suite(args)
    if "ablations" in args.mode:
        run_ablation_suites(args)
    save_history({"runs": collect_metric_summaries(METRICS_DIR)}, METRICS_DIR / "experiment_summary.json")


if __name__ == "__main__":
    main()
