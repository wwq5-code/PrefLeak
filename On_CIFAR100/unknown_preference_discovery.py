"""Semantic-blind clustering attack over anonymous CLIP retrieval outputs.

The attacker stage reads only each run's ``attacker_view`` directory. It sees
anonymous model IDs, opaque public item IDs, text queries, and paired top-k
rankings from M0 and Ms. It does not read preference labels, image styles,
similarity scores, embeddings, checkpoints, or training logs.

Ground-truth preferences are loaded only after clustering finishes and are used
solely for permutation-invariant evaluation (accuracy, ARI, and NMI).
"""

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cluster private preferences from anonymous paired top-k outputs."
    )
    parser.add_argument(
        "--input_glob",
        "--input-glob",
        default="On_CIFAR100/outputs/prefleak_clip_unknown_seed*",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="On_CIFAR100/outputs/prefleak_clip_unknown_discovery",
    )
    parser.add_argument("--max_iterations", "--max-iterations", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self_test", "--self-test", action="store_true")
    return parser.parse_args(argv)


def prepare_output_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path) and os.listdir(path) and not overwrite:
        raise FileExistsError(
            "Output directory is not empty: {}. Pass --overwrite to replace reports.".format(
                path
            )
        )
    os.makedirs(path, exist_ok=True)


def write_json(path: str, value) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ordered_rows(rows: Sequence[dict]) -> List[dict]:
    rows = sorted(rows, key=lambda item: item["query_index"])
    expected = list(range(len(rows)))
    observed = [item["query_index"] for item in rows]
    if observed != expected:
        raise ValueError("Query indices are incomplete or duplicated: {}".format(observed))
    return rows


def load_attacker_view(input_dirs: Sequence[str]):
    """Load attacker-visible files only; no private file is opened here."""
    common_bank_ids = None
    common_queries = None
    samples = []
    for directory in input_dirs:
        attacker_dir = os.path.join(directory, "attacker_view")
        bank_path = os.path.join(attacker_dir, "public_bank_ids.json")
        output_path = os.path.join(attacker_dir, "retrieval_outputs.jsonl")
        bank_records = json.load(open(bank_path, "r", encoding="utf-8"))
        bank_ids = [item["item_id"] for item in bank_records]
        if any(sorted(item) != ["item_id"] for item in bank_records):
            raise ValueError("Attacker bank file contains non-ID metadata: {}".format(bank_path))
        if common_bank_ids is None:
            common_bank_ids = bank_ids
        elif common_bank_ids != bank_ids:
            raise ValueError("Public bank IDs differ across model seeds.")

        grouped = defaultdict(list)
        for record in read_jsonl(output_path):
            allowed = {"model_version", "query_index", "query", "top_k_item_ids"}
            if set(record) != allowed:
                raise ValueError("Unexpected attacker-visible field(s): {}".format(record))
            grouped[record["model_version"]].append(record)
        if "M0_reference" not in grouped:
            raise ValueError("Missing M0_reference in {}".format(output_path))
        reference_rows = ordered_rows(grouped["M0_reference"])
        queries = [item["query"] for item in reference_rows]
        if common_queries is None:
            common_queries = queries
        elif common_queries != queries:
            raise ValueError("Text queries differ across model seeds.")

        target_ids = sorted(key for key in grouped if key.startswith("target_"))
        if not target_ids:
            raise ValueError("No anonymous target models in {}".format(output_path))
        run_id = os.path.basename(os.path.normpath(directory))
        for target_id in target_ids:
            target_rows = ordered_rows(grouped[target_id])
            if [item["query"] for item in target_rows] != queries:
                raise ValueError("Target/reference query mismatch for {}.".format(target_id))
            samples.append(
                {
                    "anonymous_instance_id": "{}/{}".format(run_id, target_id),
                    "run_directory": directory,
                    "anonymous_model_id": target_id,
                    "reference_rows": reference_rows,
                    "target_rows": target_rows,
                }
            )
    return common_bank_ids, common_queries, samples


def rank_delta_feature(
    reference_rows: Sequence[dict],
    target_rows: Sequence[dict],
    item_to_index: Dict[str, int],
) -> torch.Tensor:
    """Rank-weighted target-minus-reference vector with no semantic features."""
    if len(reference_rows) != len(target_rows):
        raise ValueError("Reference and target query counts differ.")
    top_k = len(reference_rows[0]["top_k_item_ids"])
    weights = torch.tensor(
        [1.0 / math.log2(rank + 2.0) for rank in range(top_k)],
        dtype=torch.float32,
    )
    feature = torch.zeros(len(reference_rows), len(item_to_index), dtype=torch.float32)
    for query_index, (reference, target) in enumerate(zip(reference_rows, target_rows)):
        reference_ids = reference["top_k_item_ids"]
        target_ids = target["top_k_item_ids"]
        if len(reference_ids) != top_k or len(target_ids) != top_k:
            raise ValueError("Inconsistent top-k length.")
        for rank, item_id in enumerate(reference_ids):
            feature[query_index, item_to_index[item_id]] -= weights[rank]
        for rank, item_id in enumerate(target_ids):
            feature[query_index, item_to_index[item_id]] += weights[rank]
    feature = feature.flatten()
    norm = feature.norm()
    if float(norm) <= 0.0:
        raise ValueError("A target has no observable ranking change.")
    return feature / norm


def build_feature_matrix(bank_ids: Sequence[str], samples: Sequence[dict]) -> torch.Tensor:
    item_to_index = {item_id: index for index, item_id in enumerate(bank_ids)}
    if len(item_to_index) != len(bank_ids):
        raise ValueError("Public bank IDs are not unique.")
    features = [
        rank_delta_feature(
            sample["reference_rows"], sample["target_rows"], item_to_index
        )
        for sample in samples
    ]
    return torch.stack(features, dim=0)


def kmeans_two(features: torch.Tensor, max_iterations: int = 100):
    """Deterministic two-means with farthest-pair initialization."""
    if features.ndim != 2 or features.shape[0] < 4:
        raise ValueError("At least four feature vectors are required for clustering.")
    pairwise = torch.cdist(features, features, p=2)
    flat_index = int(pairwise.argmax())
    first = flat_index // features.shape[0]
    second = flat_index % features.shape[0]
    centroids = torch.stack([features[first], features[second]], dim=0)
    assignments = None
    for iteration in range(1, max_iterations + 1):
        distances = torch.cdist(features, centroids, p=2)
        new_assignments = distances.argmin(dim=1)
        if assignments is not None and torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        cluster_sizes = torch.bincount(assignments, minlength=2)
        if int(cluster_sizes.min()) == 0:
            raise RuntimeError("Two-means produced an empty cluster.")
        centroids = torch.stack(
            [features[assignments == cluster].mean(dim=0) for cluster in range(2)],
            dim=0,
        )
    inertia = float(
        sum(
            (features[index] - centroids[int(assignments[index])]).square().sum()
            for index in range(features.shape[0])
        )
    )
    return assignments, centroids, iteration, inertia


def silhouette_score(features: torch.Tensor, assignments: torch.Tensor) -> float:
    distances = torch.cdist(features, features, p=2)
    scores = []
    for index in range(features.shape[0]):
        own = int(assignments[index])
        same = (assignments == own).nonzero(as_tuple=False).flatten()
        same = same[same != index]
        other = (assignments != own).nonzero(as_tuple=False).flatten()
        if same.numel() == 0:
            scores.append(0.0)
            continue
        within = float(distances[index, same].mean())
        between = float(distances[index, other].mean())
        scores.append((between - within) / max(within, between, 1e-30))
    return sum(scores) / len(scores)


def pca_first_component_fraction(features: torch.Tensor) -> float:
    centered = features - features.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    return float(eigenvalues[-1] / eigenvalues.sum().clamp_min(1e-30))


def load_private_ground_truth(input_dirs: Sequence[str]) -> Dict[str, str]:
    """Evaluation-only stage, deliberately separate from attacker loading."""
    truth = {}
    for directory in input_dirs:
        private_path = os.path.join(directory, "private_evaluation.json")
        private = json.load(open(private_path, "r", encoding="utf-8"))
        run_id = os.path.basename(os.path.normpath(directory))
        for target in private["targets"]:
            instance_id = "{}/{}".format(run_id, target["anonymous_model_id"])
            truth[instance_id] = target["preference"]
    return truth


def combination_two(value: int) -> float:
    return value * (value - 1) / 2.0


def adjusted_rand_index(labels: Sequence[int], clusters: Sequence[int]) -> float:
    table = Counter(zip(labels, clusters))
    label_counts = Counter(labels)
    cluster_counts = Counter(clusters)
    total_pairs = combination_two(len(labels))
    index = sum(combination_two(value) for value in table.values())
    label_pairs = sum(combination_two(value) for value in label_counts.values())
    cluster_pairs = sum(combination_two(value) for value in cluster_counts.values())
    expected = label_pairs * cluster_pairs / max(total_pairs, 1e-30)
    maximum = 0.5 * (label_pairs + cluster_pairs)
    denominator = maximum - expected
    return (index - expected) / denominator if abs(denominator) > 1e-30 else 1.0


def normalized_mutual_information(labels: Sequence[int], clusters: Sequence[int]) -> float:
    total = len(labels)
    table = Counter(zip(labels, clusters))
    label_counts = Counter(labels)
    cluster_counts = Counter(clusters)
    mutual_information = 0.0
    for (label, cluster), count in table.items():
        mutual_information += (count / total) * math.log(
            count * total / (label_counts[label] * cluster_counts[cluster])
        )
    label_entropy = -sum(
        (count / total) * math.log(count / total) for count in label_counts.values()
    )
    cluster_entropy = -sum(
        (count / total) * math.log(count / total) for count in cluster_counts.values()
    )
    return mutual_information / max(math.sqrt(label_entropy * cluster_entropy), 1e-30)


def evaluate_clusters(
    samples: Sequence[dict],
    assignments: torch.Tensor,
    features: torch.Tensor,
    ground_truth: Dict[str, str],
):
    semantic_values = sorted(set(ground_truth.values()))
    if len(semantic_values) != 2:
        raise ValueError("Expected exactly two hidden preference classes.")
    semantic_to_int = {value: index for index, value in enumerate(semantic_values)}
    labels = [
        semantic_to_int[ground_truth[sample["anonymous_instance_id"]]] for sample in samples
    ]
    clusters = [int(value) for value in assignments]
    direct_matches = sum(int(label == cluster) for label, cluster in zip(labels, clusters))
    best_accuracy = max(direct_matches, len(labels) - direct_matches) / len(labels)

    pair_correct = 0
    pair_total = 0
    same_similarities = []
    different_similarities = []
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            same_label = labels[first] == labels[second]
            same_cluster = clusters[first] == clusters[second]
            pair_correct += int(same_label == same_cluster)
            pair_total += 1
            similarity = float(F.cosine_similarity(features[first], features[second], dim=0))
            (same_similarities if same_label else different_similarities).append(similarity)

    composition = {}
    for cluster in range(2):
        composition[str(cluster)] = dict(
            Counter(
                ground_truth[sample["anonymous_instance_id"]]
                for sample, assignment in zip(samples, clusters)
                if assignment == cluster
            )
        )
    return {
        "best_permutation_accuracy": best_accuracy,
        "adjusted_rand_index": adjusted_rand_index(labels, clusters),
        "normalized_mutual_information": normalized_mutual_information(labels, clusters),
        "pairwise_same_different_accuracy": pair_correct / pair_total,
        "mean_cosine_similarity_same_preference": sum(same_similarities)
        / len(same_similarities),
        "mean_cosine_similarity_different_preference": sum(different_similarities)
        / len(different_similarities),
        "cluster_composition_private": composition,
        "ground_truth_private": {
            sample["anonymous_instance_id"]: ground_truth[sample["anonymous_instance_id"]]
            for sample in samples
        },
    }


def run_self_test() -> None:
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.1, 0.0],
                [0.9, -0.1, 0.1],
                [1.1, 0.0, -0.1],
                [-1.0, 0.0, 0.1],
                [-0.9, 0.1, -0.1],
                [-1.1, -0.1, 0.0],
            ]
        ),
        dim=-1,
    )
    assignments, _, _, _ = kmeans_two(features)
    samples = [{"anonymous_instance_id": "x/{}".format(i)} for i in range(6)]
    truth = {"x/{}".format(i): "alpha" if i < 3 else "beta" for i in range(6)}
    evaluation = evaluate_clusters(samples, assignments, features, truth)
    assert evaluation["best_permutation_accuracy"] == 1.0
    assert abs(evaluation["adjusted_rand_index"] - 1.0) < 1e-12
    assert abs(evaluation["normalized_mutual_information"] - 1.0) < 1e-12
    print("Self-test passed: semantic-blind clustering is permutation invariant.")


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    input_dirs = sorted(path for path in glob.glob(args.input_glob) if os.path.isdir(path))
    if len(input_dirs) < 2:
        raise ValueError("Need at least two run directories; found {}.".format(len(input_dirs)))
    prepare_output_dir(args.output_dir, args.overwrite)

    # Stage 1: attacker-only computation. No private file has been read yet.
    bank_ids, queries, samples = load_attacker_view(input_dirs)
    features = build_feature_matrix(bank_ids, samples)
    assignments, _, iterations, inertia = kmeans_two(features, args.max_iterations)
    attacker_result = {
        "attack": "semantic-blind k=2 clustering of paired rank-delta vectors",
        "uses_preference_labels": False,
        "uses_style_metadata": False,
        "uses_scores_or_embeddings": False,
        "num_run_directories": len(input_dirs),
        "num_target_instances": len(samples),
        "num_queries": len(queries),
        "public_bank_size": len(bank_ids),
        "feature_dimension": int(features.shape[1]),
        "iterations": iterations,
        "inertia": inertia,
        "silhouette_score": silhouette_score(features, assignments),
        "pca_first_component_variance_fraction": pca_first_component_fraction(features),
        "cluster_sizes": dict(Counter(str(int(value)) for value in assignments)),
        "assignments": [
            {
                "anonymous_instance_id": sample["anonymous_instance_id"],
                "cluster": int(assignment),
            }
            for sample, assignment in zip(samples, assignments)
        ],
    }
    write_json(os.path.join(args.output_dir, "attacker_cluster_assignments.json"), attacker_result)

    # Stage 2: post-hoc evaluation only. Cluster assignments are already fixed.
    ground_truth = load_private_ground_truth(input_dirs)
    private_evaluation = evaluate_clusters(samples, assignments, features, ground_truth)
    private_evaluation.update(
        {
            "attacker_result_file": "attacker_cluster_assignments.json",
            "evaluation_note": (
                "Hidden preference labels were loaded only after clustering and were "
                "used only for permutation-invariant evaluation."
            ),
        }
    )
    write_json(os.path.join(args.output_dir, "private_cluster_evaluation.json"), private_evaluation)

    report = """# Unknown-preference discovery

The attacker clustered {num_targets} anonymous target models from {num_runs}
independent training runs. It used {num_queries} text queries and paired top-k
ID changes over a fixed bank of {bank_size} opaque items. It received no
warm/cool semantics, style metadata, similarity scores, embeddings, model
parameters, or target labels.

## Unsupervised diagnostics

- Cluster sizes: {cluster_sizes}
- Silhouette score: {silhouette:.6f}
- First PCA component variance fraction: {pca_fraction:.6f}

## Post-hoc private evaluation

- Best-permutation clustering accuracy: {accuracy:.6f}
- Adjusted Rand index: {ari:.6f}
- Normalized mutual information: {nmi:.6f}
- Pairwise same/different accuracy: {pair_accuracy:.6f}
- Mean same-preference cosine similarity: {same_similarity:.6f}
- Mean different-preference cosine similarity: {different_similarity:.6f}

Cluster numbers have no semantic meaning. Hidden preference labels were loaded
only after the assignments had been written, and were used only to score the
partition up to label permutation.
""".format(
        num_targets=len(samples),
        num_runs=len(input_dirs),
        num_queries=len(queries),
        bank_size=len(bank_ids),
        cluster_sizes=attacker_result["cluster_sizes"],
        silhouette=attacker_result["silhouette_score"],
        pca_fraction=attacker_result["pca_first_component_variance_fraction"],
        accuracy=private_evaluation["best_permutation_accuracy"],
        ari=private_evaluation["adjusted_rand_index"],
        nmi=private_evaluation["normalized_mutual_information"],
        pair_accuracy=private_evaluation["pairwise_same_different_accuracy"],
        same_similarity=private_evaluation["mean_cosine_similarity_same_preference"],
        different_similarity=private_evaluation[
            "mean_cosine_similarity_different_preference"
        ],
    )
    with open(os.path.join(args.output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)
    print("Results written to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
