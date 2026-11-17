"""CLIP-based PrefLeak pre-experiment on CIFAR-100.

This script intentionally studies a *retrieval* variant of PrefLeak rather than
claiming that CLIP is a generative VLM. A private preference update teaches a
CLIP model to prefer either warm or cool versions of otherwise identical
CIFAR-100 images. The attacker then sends text-only queries to a fixed public
image bank and observes only the returned top-k string identifiers.

The paired-version attack compares retrieval outputs from the public reference
model M0 and a privately updated model Ms. It never consumes similarities,
embeddings, gradients, or model parameters. A logit-scale control is trained
with the same preference objective; because a global scale cannot change a
top-k ranking, that control should produce no observable leakage.

This is a mechanism/feasibility experiment for PrefLeak. The main paper's
sampled-text threat model still requires a generative VLM experiment.
"""

import argparse
import json
import math
import os
import random
import secrets
from typing import Dict, Iterable, List, Sequence, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import functional as TF


PREFERENCES = ("warm", "cool")
STYLE_SIGN = {"warm": 1.0, "cool": -1.0}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Text-only retrieval PrefLeak pre-experiment with OpenCLIP."
    )
    parser.add_argument("--data_root", "--data-root", default="data/cifar")
    parser.add_argument("--cache_dir", "--cache-dir", default="models/open_clip")
    parser.add_argument("--model_name", "--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--pretrained_path",
        "--pretrained-path",
        default=None,
        help="Local OpenCLIP checkpoint; preferred when Hugging Face is unavailable.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="On_CIFAR100/outputs/prefleak_clip_preexperiment",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--attack_mode",
        "--attack-mode",
        choices=("unknown_discovery", "known_direction"),
        default="unknown_discovery",
        help=(
            "unknown_discovery exports only anonymous rankings; known_direction "
            "retains the earlier style-aware diagnostic attack."
        ),
    )
    parser.add_argument(
        "--bank_seed",
        "--bank-seed",
        type=int,
        default=314159,
        help="Fixed across model seeds so anonymous rank-delta features are aligned.",
    )
    parser.add_argument(
        "--preference",
        choices=("warm", "cool", "both"),
        default="both",
        help="Private preference(s) used to create target models.",
    )
    parser.add_argument(
        "--trainable_scope",
        "--trainable-scope",
        choices=("visual_projection", "visual"),
        default="visual_projection",
        help="Update only the final vision projection by default.",
    )
    parser.add_argument("--tint_strength", "--tint-strength", type=float, default=0.20)
    parser.add_argument("--saturation", type=float, default=1.15)
    parser.add_argument("--num_train_samples", "--num-train-samples", type=int, default=5000)
    parser.add_argument("--train_batch_size", "--train-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", "--max-steps", type=int, default=None)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--grad_clip", "--grad-clip", type=float, default=1.0)
    parser.add_argument("--num_bank_samples", "--num-bank-samples", type=int, default=1000)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=256)
    parser.add_argument("--top_k", "--top-k", type=int, default=20)
    parser.add_argument("--num_queries", "--num-queries", type=int, default=100)
    parser.add_argument(
        "--query_templates",
        "--query-templates",
        nargs="+",
        default=["an image of a {}"],
        help="Neutral text-only retrieval templates; each must contain one {}.",
    )
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--log_every", "--log-every", type=int, default=25)
    parser.add_argument(
        "--skip_noninterference_control",
        "--skip-noninterference-control",
        action="store_true",
        help="Skip the trained logit-scale control whose top-k output must stay fixed.",
    )
    parser.add_argument(
        "--save_checkpoint",
        "--save-checkpoint",
        action="store_true",
        help="Save only the parameters updated in each target model.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing named result files in a non-empty output directory.",
    )
    parser.add_argument(
        "--self_test",
        "--self-test",
        action="store_true",
        help="Run fast metric tests without loading CLIP or CIFAR-100.",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_args(args) -> None:
    if not 0.0 <= args.tint_strength <= 0.75:
        raise ValueError("--tint-strength must lie in [0, 0.75].")
    if args.saturation <= 0.0:
        raise ValueError("--saturation must be positive.")
    for name in (
        "num_train_samples",
        "train_batch_size",
        "epochs",
        "num_bank_samples",
        "eval_batch_size",
        "top_k",
        "num_queries",
    ):
        if getattr(args, name) <= 0:
            raise ValueError("--{} must be positive.".format(name.replace("_", "-")))
    if args.beta <= 0.0:
        raise ValueError("--beta must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive when set.")
    if args.attack_mode == "unknown_discovery" and args.preference != "both":
        raise ValueError("Unknown-preference discovery requires --preference both.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is false.")
    for template in args.query_templates:
        try:
            template.format("example")
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError("Invalid query template {!r}: {}".format(template, error))
        if "{}" not in template:
            raise ValueError("Each query template must contain one plain {} placeholder.")


def prepare_output_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path) and os.listdir(path) and not overwrite:
        raise FileExistsError(
            "Output directory is not empty: {}. Pass --overwrite or choose another path.".format(
                path
            )
        )
    os.makedirs(path, exist_ok=True)


def write_json(path: str, value) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: str, records: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def resolve_pretrained(args):
    if args.pretrained_path is not None:
        if not os.path.isfile(args.pretrained_path):
            raise FileNotFoundError("Checkpoint not found: {}".format(args.pretrained_path))
        return args.pretrained_path
    if os.path.isfile(args.pretrained):
        return args.pretrained
    local_checkpoint = os.path.join(args.cache_dir, "{}.pt".format(args.model_name))
    if os.path.isfile(local_checkpoint):
        return local_checkpoint
    return args.pretrained


def load_checkpoint_flexible(model: nn.Module, checkpoint_path: str) -> None:
    try:
        checkpoint = torch.jit.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.state_dict()
        print("Loaded TorchScript checkpoint: {}".format(checkpoint_path))
    except RuntimeError:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        elif hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()
        else:
            raise RuntimeError("Unsupported checkpoint format: {}".format(checkpoint_path))
    state_dict = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print("Missing checkpoint keys: {}".format(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        print("Unexpected checkpoint keys: {}".format(incompatible.unexpected_keys))


def create_clip(args):
    pretrained = resolve_pretrained(args)
    if os.path.isfile(pretrained):
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, pretrained=None, cache_dir=args.cache_dir
        )
        load_checkpoint_flexible(model, pretrained)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, pretrained=pretrained, cache_dir=args.cache_dir
        )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    return model, preprocess, tokenizer, pretrained


def tint_image(image, style: str, strength: float, saturation: float):
    """Create deterministic warm/cool views without changing image geometry."""
    if style not in PREFERENCES:
        raise ValueError("Unknown style: {}".format(style))
    tensor = TF.pil_to_tensor(image).float().div_(255.0)
    gray = (
        0.2989 * tensor[0:1]
        + 0.5870 * tensor[1:2]
        + 0.1140 * tensor[2:3]
    )
    tensor = gray + saturation * (tensor - gray)
    if style == "warm":
        gains = tensor.new_tensor([1.0 + strength, 1.0, 1.0 - strength])
    else:
        gains = tensor.new_tensor([1.0 - strength, 1.0, 1.0 + strength])
    tensor = (tensor * gains[:, None, None]).clamp_(0.0, 1.0)
    return TF.to_pil_image(tensor)


class PreferencePairDataset(Dataset):
    """Paired views of one image; semantics and geometry are held fixed."""

    def __init__(
        self,
        base_dataset,
        indices: Sequence[int],
        preprocess,
        tint_strength: float,
        saturation: float,
    ):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.preprocess = preprocess
        self.tint_strength = tint_strength
        self.saturation = saturation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        original_index = self.indices[item]
        image, label = self.base_dataset[original_index]
        warm = tint_image(image, "warm", self.tint_strength, self.saturation)
        cool = tint_image(image, "cool", self.tint_strength, self.saturation)
        return (
            self.preprocess(warm),
            self.preprocess(cool),
            int(label),
            int(item),
            int(original_index),
        )


def choose_indices(length: int, count: int, seed: int) -> List[int]:
    if count > length:
        raise ValueError("Requested {} samples from a dataset of length {}.".format(count, length))
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def make_loader(dataset, batch_size: int, shuffle: bool, args, seed_offset: int = 0):
    generator = torch.Generator()
    generator.manual_seed(args.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


@torch.no_grad()
def encode_texts(model, tokenizer, prompts: Sequence[str], device: torch.device):
    tokens = tokenizer(list(prompts)).to(device)
    features = model.encode_text(tokens)
    return F.normalize(features.float(), dim=-1)


def class_prompts(class_names: Sequence[str]) -> List[str]:
    return ["a photo of a {}".format(name.replace("_", " ")) for name in class_names]


def build_queries(
    class_names: Sequence[str], templates: Sequence[str], num_queries: int
) -> Tuple[List[str], List[int]]:
    candidates = []
    for template in templates:
        for class_index, name in enumerate(class_names):
            candidates.append((template.format(name.replace("_", " ")), class_index))
    if num_queries > len(candidates):
        raise ValueError(
            "--num-queries={} exceeds {} template/class combinations.".format(
                num_queries, len(candidates)
            )
        )
    selected = candidates[:num_queries]
    return [item[0] for item in selected], [item[1] for item in selected]


def model_scale(model) -> torch.Tensor:
    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is None:
        raise AttributeError("The OpenCLIP model has no logit_scale parameter.")
    # Do not clamp here. OpenAI ViT-B/32 starts at (or microscopically above)
    # exp(logit_scale)=100, so a max=100 clamp gives the non-interference
    # control exactly zero gradient. An unconstrained positive global scale can
    # learn under the preference loss while still being unable to change a
    # retrieval ranking.
    return logit_scale.exp()


def configure_trainable(model, scope: str) -> List[Tuple[str, nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if scope == "visual":
        for parameter in model.visual.parameters():
            parameter.requires_grad_(True)
    elif scope == "visual_projection":
        projection = getattr(model.visual, "proj", None)
        if isinstance(projection, nn.Parameter):
            projection.requires_grad_(True)
        elif isinstance(projection, nn.Module):
            for parameter in projection.parameters():
                parameter.requires_grad_(True)
        else:
            raise TypeError(
                "visual_projection requires visual.proj to be a Parameter or Module."
            )
    elif scope == "logit_scale_control":
        model.logit_scale.requires_grad_(True)
    else:
        raise ValueError("Unknown trainable scope: {}".format(scope))

    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters for scope {}.".format(scope))
    return trainable


def capture_parameters(named_parameters: Sequence[Tuple[str, nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in named_parameters}


def restore_parameters(model, state: Dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


@torch.no_grad()
def precompute_reference_margins(
    model,
    loader,
    text_features: torch.Tensor,
    dataset_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return unscaled sim(warm,text)-sim(cool,text) for every train pair."""
    margins = torch.empty(dataset_size, dtype=torch.float32)
    model.eval()
    for warm, cool, labels, local_indices, _ in loader:
        images = torch.cat([warm, cool], dim=0).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = F.normalize(model.encode_image(images).float(), dim=-1)
        warm_features, cool_features = features.chunk(2, dim=0)
        matching_text = text_features.index_select(0, labels)
        batch_margins = ((warm_features - cool_features) * matching_text).sum(dim=-1)
        margins[local_indices.long()] = batch_margins.cpu()
    if not torch.isfinite(margins).all():
        raise RuntimeError("Reference margin precomputation produced non-finite values.")
    return margins


def preference_sign(preference: str) -> float:
    return STYLE_SIGN[preference]


def train_preference_model(
    model,
    loader,
    text_features: torch.Tensor,
    reference_unscaled_margins: torch.Tensor,
    reference_scale: float,
    preference: str,
    scope: str,
    args,
    device: torch.device,
    run_name: str,
    log_path: str,
):
    trainable = configure_trainable(model, scope)
    initial_state = capture_parameters(trainable)
    optimizer = torch.optim.AdamW(
        [value for _, value in trainable],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    sign = preference_sign(preference)
    step = 0
    records = []
    model.eval()

    for epoch in range(1, args.epochs + 1):
        for warm, cool, labels, local_indices, _ in loader:
            step += 1
            images = torch.cat([warm, cool], dim=0).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            local_indices = local_indices.long()

            features = F.normalize(model.encode_image(images).float(), dim=-1)
            warm_features, cool_features = features.chunk(2, dim=0)
            matching_text = text_features.index_select(0, labels)
            unscaled_margin = ((warm_features - cool_features) * matching_text).sum(dim=-1)
            target_margin = sign * model_scale(model) * unscaled_margin
            reference_margin = (
                sign
                * reference_scale
                * reference_unscaled_margins.index_select(0, local_indices).to(device)
            )
            advantage = target_margin - reference_margin
            loss = F.softplus(-args.beta * advantage).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [value for _, value in trainable], args.grad_clip
                )
            optimizer.step()

            if step == 1 or step % args.log_every == 0:
                record = {
                    "run": run_name,
                    "scope": scope,
                    "preference": preference,
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "mean_advantage": float(advantage.detach().mean().cpu()),
                    "batch_preference_accuracy": float(
                        target_margin.detach().gt(0).float().mean().cpu()
                    ),
                    "logit_scale": float(model_scale(model).detach().cpu()),
                }
                records.append(record)
                print(
                    "[{run}] epoch={epoch} step={step} loss={loss:.6f} "
                    "adv={adv:.6f} pref_acc={acc:.3f}".format(
                        run=run_name,
                        epoch=epoch,
                        step=step,
                        loss=record["loss"],
                        adv=record["mean_advantage"],
                        acc=record["batch_preference_accuracy"],
                    )
                )
            if args.max_steps is not None and step >= args.max_steps:
                break
        if args.max_steps is not None and step >= args.max_steps:
            break

    append_jsonl(log_path, records)
    final_state = capture_parameters(trainable)
    delta_sq = 0.0
    reference_sq = 0.0
    for name in initial_state:
        delta = final_state[name].float() - initial_state[name].float()
        delta_sq += float(delta.square().sum())
        reference_sq += float(initial_state[name].float().square().sum())
    diagnostics = {
        "steps": step,
        "trainable_parameter_count": sum(value.numel() for _, value in trainable),
        "parameter_delta_l2": math.sqrt(delta_sq),
        "relative_parameter_delta": math.sqrt(delta_sq) / max(math.sqrt(reference_sq), 1e-30),
        "final_logit_scale": float(model_scale(model).detach().cpu()),
    }
    return initial_state, final_state, diagnostics


@torch.no_grad()
def encode_public_bank(model, loader, device: torch.device):
    warm_features = []
    cool_features = []
    labels = []
    original_indices = []
    model.eval()
    for warm, cool, batch_labels, _, batch_original_indices in loader:
        images = torch.cat([warm, cool], dim=0).to(device, non_blocking=True)
        features = F.normalize(model.encode_image(images).float(), dim=-1)
        warm_batch, cool_batch = features.chunk(2, dim=0)
        warm_features.append(warm_batch.cpu())
        cool_features.append(cool_batch.cpu())
        labels.extend(int(value) for value in batch_labels)
        original_indices.extend(int(value) for value in batch_original_indices)
    warm_features = torch.cat(warm_features, dim=0)
    cool_features = torch.cat(cool_features, dim=0)
    return warm_features, cool_features, labels, original_indices


def bank_metadata(
    labels: Sequence[int], original_indices: Sequence[int], class_names: Sequence[str]
):
    records = []
    for style in PREFERENCES:
        suffix = "a" if style == "warm" else "b"
        for label, original_index in zip(labels, original_indices):
            records.append(
                {
                    "item_id": "cifar100-test-{:05d}-{}".format(original_index, suffix),
                    "cifar_index": original_index,
                    "class_name": class_names[label],
                    "style": style,
                }
            )
    return records


def anonymize_bank(metadata: Sequence[dict], seed: int):
    """Shuffle the bank and replace style-bearing identifiers with opaque IDs."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(metadata), generator=generator)
    private_records = []
    public_records = []
    for public_index, private_index in enumerate(permutation.tolist()):
        private_record = dict(metadata[private_index])
        opaque_id = "public-item-{:05d}".format(public_index)
        private_record["item_id"] = opaque_id
        private_records.append(private_record)
        public_records.append({"item_id": opaque_id})
    return permutation, private_records, public_records


def topk_retrieval(text_features: torch.Tensor, bank_features: torch.Tensor, top_k: int):
    if top_k > bank_features.shape[0]:
        raise ValueError("top-k exceeds the public bank size.")
    similarities = text_features.cpu().float() @ bank_features.cpu().float().T
    return similarities.topk(top_k, dim=-1, largest=True, sorted=True).indices


def retrieval_records(
    model_version: str,
    prompts: Sequence[str],
    indices: torch.Tensor,
    metadata: Sequence[dict],
):
    records = []
    for query_index, (prompt, row) in enumerate(zip(prompts, indices.tolist())):
        records.append(
            {
                "model_version": model_version,
                "query_index": query_index,
                "query": prompt,
                "top_k_item_ids": [metadata[item]["item_id"] for item in row],
            }
        )
    return records


def compute_attack_metrics(
    reference_indices: torch.Tensor,
    target_indices: torch.Tensor,
    style_signs: torch.Tensor,
    true_preference: str,
):
    """Attack using returned IDs plus the public bank manifest, never scores."""
    if reference_indices.shape != target_indices.shape:
        raise ValueError("Reference and target top-k arrays must have the same shape.")
    reference_style = style_signs[reference_indices].float().mean(dim=-1)
    target_style = style_signs[target_indices].float().mean(dim=-1)
    query_drifts = target_style - reference_style

    overlaps = []
    exact_changes = []
    for reference_row, target_row in zip(reference_indices.tolist(), target_indices.tolist()):
        overlaps.append(len(set(reference_row).intersection(target_row)) / len(reference_row))
        exact_changes.append(float(reference_row != target_row))

    drift = float(query_drifts.mean())
    if abs(drift) <= 1e-12:
        predicted = "unknown"
    else:
        predicted = "warm" if drift > 0.0 else "cool"
    return {
        "true_preference": true_preference,
        "predicted_preference": predicted,
        "correct": bool(predicted == true_preference),
        "reference_mean_style": float(reference_style.mean()),
        "target_mean_style": float(target_style.mean()),
        "style_drift": drift,
        "mean_absolute_query_drift": float(query_drifts.abs().mean()),
        "top_k_overlap": float(sum(overlaps) / len(overlaps)),
        "response_change_rate": float(sum(exact_changes) / len(exact_changes)),
        "num_queries": int(reference_indices.shape[0]),
        "top_k": int(reference_indices.shape[1]),
    }


def compute_output_change_metrics(
    reference_indices: torch.Tensor,
    target_indices: torch.Tensor,
):
    """Semantic-free observables computed from two top-k ID lists."""
    if reference_indices.shape != target_indices.shape:
        raise ValueError("Reference and target top-k arrays must have the same shape.")
    overlaps = []
    exact_changes = []
    for reference_row, target_row in zip(reference_indices.tolist(), target_indices.tolist()):
        overlaps.append(len(set(reference_row).intersection(target_row)) / len(reference_row))
        exact_changes.append(float(reference_row != target_row))
    return {
        "top_k_overlap": float(sum(overlaps) / len(overlaps)),
        "response_change_rate": float(sum(exact_changes) / len(exact_changes)),
        "num_queries": int(reference_indices.shape[0]),
        "top_k": int(reference_indices.shape[1]),
    }


def preference_eval(
    warm_features: torch.Tensor,
    cool_features: torch.Tensor,
    labels: Sequence[int],
    class_text_features: torch.Tensor,
    preference: str,
):
    label_tensor = torch.tensor(labels, dtype=torch.long)
    matching_text = class_text_features.cpu().index_select(0, label_tensor)
    warm_minus_cool = ((warm_features - cool_features) * matching_text).sum(dim=-1)
    signed_margin = preference_sign(preference) * warm_minus_cool
    return {
        "pairwise_preference_accuracy": float(signed_margin.gt(0).float().mean()),
        "mean_signed_cosine_margin": float(signed_margin.mean()),
    }


def run_self_test() -> None:
    style_signs = torch.tensor([1.0, 1.0, -1.0, -1.0])
    reference = torch.tensor([[0, 2], [1, 3]])
    warm_target = torch.tensor([[0, 1], [1, 0]])
    cool_target = torch.tensor([[2, 3], [3, 2]])
    warm = compute_attack_metrics(reference, warm_target, style_signs, "warm")
    cool = compute_attack_metrics(reference, cool_target, style_signs, "cool")
    unchanged = compute_attack_metrics(reference, reference.clone(), style_signs, "warm")
    assert warm["predicted_preference"] == "warm" and warm["correct"]
    assert cool["predicted_preference"] == "cool" and cool["correct"]
    assert unchanged["predicted_preference"] == "unknown"
    assert unchanged["top_k_overlap"] == 1.0
    assert unchanged["response_change_rate"] == 0.0
    print("Self-test passed: paired-ID attack and non-interference behavior are correct.")


def selected_preferences(argument: str) -> List[str]:
    return list(PREFERENCES) if argument == "both" else [argument]


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    validate_args(args)
    set_seed(args.seed)
    prepare_output_dir(args.output_dir, args.overwrite)

    device = torch.device(args.device)
    print("Loading OpenCLIP {} ({})...".format(args.model_name, args.pretrained))
    model, preprocess, tokenizer, resolved_pretrained = create_clip(args)
    model = model.to(device)
    model.eval()

    train_base = CIFAR100(root=args.data_root, train=True, download=args.download)
    test_base = CIFAR100(root=args.data_root, train=False, download=args.download)
    if train_base.classes != test_base.classes:
        raise RuntimeError("CIFAR-100 train/test class lists differ.")
    train_indices = choose_indices(len(train_base), args.num_train_samples, args.seed + 11)
    # The train split changes with the model seed. The public bank is fixed
    # across model seeds so anonymous output-delta features share coordinates.
    bank_indices = choose_indices(len(test_base), args.num_bank_samples, args.bank_seed)
    train_dataset = PreferencePairDataset(
        train_base, train_indices, preprocess, args.tint_strength, args.saturation
    )
    bank_dataset = PreferencePairDataset(
        test_base, bank_indices, preprocess, args.tint_strength, args.saturation
    )
    reference_loader = make_loader(
        train_dataset, args.eval_batch_size, False, args, seed_offset=101
    )
    bank_loader = make_loader(
        bank_dataset, args.eval_batch_size, False, args, seed_offset=307
    )

    class_text_features = encode_texts(
        model, tokenizer, class_prompts(train_base.classes), device
    )
    query_prompts, _ = build_queries(
        train_base.classes, args.query_templates, args.num_queries
    )
    query_text_features = encode_texts(model, tokenizer, query_prompts, device)
    reference_scale = float(model_scale(model).detach().cpu())

    print("Precomputing reference preference margins...")
    reference_margins = precompute_reference_margins(
        model,
        reference_loader,
        class_text_features,
        len(train_dataset),
        device,
    )
    print("Encoding the fixed public retrieval bank...")
    reference_warm, reference_cool, bank_labels, bank_original_indices = encode_public_bank(
        model, bank_loader, device
    )
    reference_bank_unshuffled = torch.cat([reference_warm, reference_cool], dim=0)
    raw_metadata = bank_metadata(bank_labels, bank_original_indices, test_base.classes)
    bank_permutation, metadata, public_metadata = anonymize_bank(
        raw_metadata, args.bank_seed + 1
    )
    reference_bank = reference_bank_unshuffled.index_select(0, bank_permutation)
    style_signs = torch.tensor(
        [STYLE_SIGN[item["style"]] for item in metadata], dtype=torch.float32
    )
    reference_topk = topk_retrieval(query_text_features, reference_bank, args.top_k)

    config = vars(args).copy()
    config.update(
        {
            "resolved_pretrained": resolved_pretrained,
            "threat_model": "text queries; paired top-k string IDs; no scores/embeddings",
            "experiment_scope": "CLIP retrieval pre-experiment, not generative VLM",
            "attacker_knowledge": (
                "anonymous model IDs and opaque top-k item IDs; no preference semantics"
                if args.attack_mode == "unknown_discovery"
                else "known warm/cool candidate semantics"
            ),
            "reference_logit_scale": reference_scale,
        }
    )
    attacker_dir = os.path.join(args.output_dir, "attacker_view")
    os.makedirs(attacker_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "config.json"), config)
    write_json(os.path.join(args.output_dir, "train_indices.json"), train_indices)
    write_json(
        os.path.join(args.output_dir, "private_evaluation_bank_manifest.json"),
        metadata,
    )
    write_json(os.path.join(attacker_dir, "public_bank_ids.json"), public_metadata)
    retrieval_path = os.path.join(attacker_dir, "retrieval_outputs.jsonl")
    training_log_path = os.path.join(args.output_dir, "training_log.jsonl")
    for path in (retrieval_path, training_log_path):
        with open(path, "w", encoding="utf-8"):
            pass
    append_jsonl(
        retrieval_path,
        retrieval_records("M0_reference", query_prompts, reference_topk, public_metadata),
    )

    results = {
        "experiment": "CLIP retrieval PrefLeak {}".format(args.attack_mode),
        "threat_model": config["threat_model"],
        "reference_pairwise_baselines": {
            preference: preference_eval(
                reference_warm,
                reference_cool,
                bank_labels,
                class_text_features,
                preference,
            )
            for preference in PREFERENCES
        },
        "targets": [],
        "noninterference_controls": [],
    }

    def run_one(preference: str, scope: str, kind: str, anonymous_model_id: str):
        run_name = "{}_{}_{}".format(kind, preference, scope)
        print("\nTraining {}...".format(run_name))
        # The optimization order is reproducible and independent of the
        # secret-random order in which anonymous targets are emitted.
        run_loader = make_loader(
            train_dataset,
            args.train_batch_size,
            True,
            args,
            seed_offset=211 if preference == "warm" else 9973,
        )
        initial_state, final_state, train_diagnostics = train_preference_model(
            model=model,
            loader=run_loader,
            text_features=class_text_features,
            reference_unscaled_margins=reference_margins,
            reference_scale=reference_scale,
            preference=preference,
            scope=scope,
            args=args,
            device=device,
            run_name=run_name,
            log_path=training_log_path,
        )
        target_warm, target_cool, target_labels, target_original_indices = encode_public_bank(
            model, bank_loader, device
        )
        if target_labels != bank_labels or target_original_indices != bank_original_indices:
            raise RuntimeError("Public bank order changed across model versions.")
        target_bank_unshuffled = torch.cat([target_warm, target_cool], dim=0)
        target_bank = target_bank_unshuffled.index_select(0, bank_permutation)
        target_topk = topk_retrieval(query_text_features, target_bank, args.top_k)
        output_change = compute_output_change_metrics(reference_topk, target_topk)
        known_direction_attack = None
        if args.attack_mode == "known_direction":
            known_direction_attack = compute_attack_metrics(
                reference_topk, target_topk, style_signs, preference
            )
        evaluation = preference_eval(
            target_warm,
            target_cool,
            bank_labels,
            class_text_features,
            preference,
        )
        record = {
            "run": run_name,
            "anonymous_model_id": anonymous_model_id,
            "kind": kind,
            "scope": scope,
            "preference": preference,
            "training": train_diagnostics,
            "heldout_preference": evaluation,
            "observable_output_change": output_change,
            "known_direction_attack": known_direction_attack,
        }
        append_jsonl(
            retrieval_path,
            retrieval_records(
                anonymous_model_id, query_prompts, target_topk, public_metadata
            ),
        )
        if args.save_checkpoint:
            torch.save(
                {
                    "run": run_name,
                    "model_name": args.model_name,
                    "scope": scope,
                    "preference": preference,
                    "trainable_state_dict": final_state,
                },
                os.path.join(args.output_dir, "{}.pt".format(run_name)),
            )
        restore_parameters(model, initial_state)
        return record

    target_index = 0
    control_index = 0
    preference_order = selected_preferences(args.preference)
    if args.attack_mode == "unknown_discovery":
        # This ordering and the opaque IDs must not be derivable from the
        # public experiment seed. They are irrelevant to model training and
        # are retained only in the private evaluator mapping.
        secrets.SystemRandom().shuffle(preference_order)
    for preference in preference_order:
        target_model_id = (
            "target_{}".format(secrets.token_hex(8))
            if args.attack_mode == "unknown_discovery"
            else "target_{:03d}".format(target_index)
        )
        results["targets"].append(
            run_one(
                preference,
                args.trainable_scope,
                "target",
                target_model_id,
            )
        )
        target_index += 1
        if not args.skip_noninterference_control:
            control_model_id = (
                "control_{}".format(secrets.token_hex(8))
                if args.attack_mode == "unknown_discovery"
                else "control_{:03d}".format(control_index)
            )
            results["noninterference_controls"].append(
                run_one(
                    preference,
                    "logit_scale_control",
                    "control",
                    control_model_id,
                )
            )
            control_index += 1

    target_changes = [item["observable_output_change"] for item in results["targets"]]
    control_changes = [
        item["observable_output_change"] for item in results["noninterference_controls"]
    ]
    results["aggregate"] = {
        "target_mean_response_change_rate": sum(
            item["response_change_rate"] for item in target_changes
        )
        / len(target_changes),
        "control_mean_response_change_rate": (
            sum(item["response_change_rate"] for item in control_changes)
            / len(control_changes)
            if control_changes
            else None
        ),
    }
    if args.attack_mode == "known_direction":
        target_attacks = [item["known_direction_attack"] for item in results["targets"]]
        results["aggregate"]["target_attack_accuracy"] = sum(
            float(item["correct"]) for item in target_attacks
        ) / len(target_attacks)
    private_results_path = os.path.join(args.output_dir, "private_evaluation.json")
    write_json(private_results_path, results)

    summary_lines = [
        "CLIP RETRIEVAL PREFLEAK {}".format(args.attack_mode.upper()),
        "Threat model: {}".format(config["threat_model"]),
    ]
    for item in results["targets"] + results["noninterference_controls"]:
        change = item["observable_output_change"]
        heldout = item["heldout_preference"]
        line = (
            "{model_id}: pref_acc(private)={pref_acc:.4f} "
            "top_k_overlap={overlap:.4f} response_change={response_change:.4f}"
        ).format(
            model_id=item["anonymous_model_id"],
            pref_acc=heldout["pairwise_preference_accuracy"],
            overlap=change["top_k_overlap"],
            response_change=change["response_change_rate"],
        )
        if item["known_direction_attack"] is not None:
            attack = item["known_direction_attack"]
            line += " prediction={} correct={}".format(
                attack["predicted_preference"], attack["correct"]
            )
        summary_lines.append(line)
    if args.attack_mode == "unknown_discovery":
        summary_lines.append(
            "No semantic prediction was made here; cluster attacker_view outputs across seeds."
        )
    else:
        summary_lines.append(
            "Aggregate target attack accuracy: {:.4f}".format(
                results["aggregate"]["target_attack_accuracy"]
            )
        )
    summary = "\n".join(summary_lines) + "\n"
    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary)
    print("\n" + summary)
    print("Results written to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
