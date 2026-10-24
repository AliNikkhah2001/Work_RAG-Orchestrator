#!/usr/bin/env python3
"""Comprehensive training and evaluation of ambiguity classifiers.

Loads train/test/eval splits, extracts features (7-dim subspace features or 384-dim
raw embeddings), trains multiple models (LR, SVM, RF, MLP, XGBoost), performs
cross-validation, evaluates on test/eval sets, runs subsample experiments for
learning curves, and generates detailed reports.

Usage:
    python scripts/train_eval_ambiguity.py \
        --train data/ambiguity_train.jsonl \
        --test data/ambiguity_test.jsonl \
        --eval data/ambiguity_eval.jsonl \
        --models lr,svm,rf,mlp,xgb \
        --output-dir data/model_eval/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_subspace_protos import (  # noqa: E402
    DEFAULT_MODEL,
    ORCH_ROOT,
    SUBSPACES,
    embed_texts,
    load_protos,
    score_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES_7D = [*SUBSPACES, "margin", "top1", "entropy"]
SUBSPACE_LIST = list(SUBSPACES)
N_SUBSPACES = len(SUBSPACES)

# Default subsample fractions for learning curves
DEFAULT_SUBSAMPLE_FRACTIONS = [0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_splits(
    train_path: Path, test_path: Path, eval_path: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load train/test/eval splits."""
    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)
    eval_rows = load_jsonl(eval_path)
    print(f"Loaded: train={len(train_rows)}, test={len(test_rows)}, eval={len(eval_rows)}")
    return train_rows, test_rows, eval_rows


def prepare_labels(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract ambiguity labels (binary) and subspace labels (multi-class)."""
    y_amb = np.array([1 if r["ambiguous"] else 0 for r in rows], dtype=int)
    y_sub = np.array([r["subspace"] if r["subspace"] else "ambiguous" for r in rows], dtype=object)
    return y_amb, y_sub


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_7d(
    rows: list[dict],
    protos_path: Path,
    model: str,
    embed_url: Optional[str] = None,
    temp: float = 0.05,
) -> np.ndarray:
    """Extract 7-dimensional subspace features (4 sims + margin + top1 + entropy)."""
    texts = [r["query"] for r in rows]
    vectors = embed_texts(texts, model=model, embed_url=embed_url)
    protos = load_protos(protos_path)
    if protos is None:
        raise RuntimeError(f"Failed to load protos from {protos_path}")
    feats = score_features(vectors, protos, temp=temp)
    # Build feature matrix in FEATURE_NAMES_7D order
    cols = [feats["sims"]]
    cols.append(feats["margin"].reshape(-1, 1))
    cols.append(feats["top1"].reshape(-1, 1))
    cols.append(feats["entropy"].reshape(-1, 1))
    X = np.concatenate(cols, axis=1).astype(np.float32)
    return X


def extract_raw_embeddings(
    rows: list[dict],
    model: str,
    embed_url: Optional[str] = None,
) -> np.ndarray:
    """Extract raw 384-dim embeddings."""
    texts = [r["query"] for r in rows]
    vectors = embed_texts(texts, model=model, embed_url=embed_url)
    return vectors.astype(np.float32)


def apply_pca(X_train: np.ndarray, X_test: np.ndarray, X_eval: np.ndarray, n_components: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Apply PCA to reduce dimensionality."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, X_train.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_eval_pca = pca.transform(X_eval)
    print(f"PCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} dims (explained variance: {pca.explained_variance_ratio_.sum():.4f})")
    return X_train_pca, X_test_pca, X_eval_pca, pca


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(model_names: list[str], random_state: int = 42, n_jobs: int = -1) -> dict[str, Any]:
    """Get dictionary of model instances."""
    models = {}
    
    if "lr" in model_names:
        from sklearn.linear_model import LogisticRegression
        models["lr"] = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        )
    
    if "svm_linear" in model_names:
        from sklearn.svm import LinearSVC
        models["svm_linear"] = LinearSVC(
            class_weight="balanced", random_state=random_state, max_iter=5000, dual="auto"
        )
    
    if "svm_rbf" in model_names:
        from sklearn.svm import SVC
        models["svm_rbf"] = SVC(
            kernel="rbf", class_weight="balanced", random_state=random_state, probability=True
        )
    
    if "rf" in model_names:
        from sklearn.ensemble import RandomForestClassifier
        models["rf"] = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=n_jobs
        )
    
    if "mlp" in model_names:
        from sklearn.neural_network import MLPClassifier
        models["mlp"] = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state, early_stopping=True
        )
    
    if "xgb" in model_names:
        try:
            from xgboost import XGBClassifier
            models["xgb"] = XGBClassifier(
                n_estimators=200, random_state=random_state, n_jobs=n_jobs,
                eval_metric="logloss", verbosity=0
            )
        except ImportError:
            print("[warn] xgboost not available, skipping")
    
    return models


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, labels: Optional[list] = None) -> dict[str, Any]:
    """Compute comprehensive classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        balanced_accuracy_score, roc_auc_score, confusion_matrix,
        classification_report
    )
    
    # Determine unique labels
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics["per_class"] = {}
    for lab in labels:
        if lab in report and isinstance(report[lab], dict):
            # Handle case where 'f1' might not be present
            metrics["per_class"][lab] = {
                "precision": float(report[lab].get("precision", 0.0)),
                "recall": float(report[lab].get("recall", 0.0)),
                "f1": float(report[lab].get("f1-score", report[lab].get("f1", 0.0))),
                "support": int(report[lab].get("support", 0)),
            }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = labels
    
    # ROC-AUC (for binary or OvR multiclass)
    if y_proba is not None:
        try:
            if len(labels) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels))
                metrics["roc_auc_ovo"] = float(roc_auc_score(y_true, y_proba, multi_class="ovo", labels=labels))
        except Exception as e:
            metrics["roc_auc_error"] = str(e)
    
    return metrics


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> dict[str, float]:
    """Compute binary classification metrics (ambiguous vs non-ambiguous)."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        balanced_accuracy_score, roc_auc_score
    )
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    if y_proba is not None and y_proba.shape[1] >= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            pass
    
    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform stratified cross-validation."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Accuracy
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    bal_acc_scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    
    # Predictions for confusion matrix
    y_pred_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    cv_metrics = compute_metrics(y, y_pred_cv)
    
    return {
        "cv_folds": cv_folds,
        "accuracy": {
            "mean": float(np.mean(acc_scores)),
            "std": float(np.std(acc_scores)),
            "scores": acc_scores.tolist(),
        },
        "balanced_accuracy": {
            "mean": float(np.mean(bal_acc_scores)),
            "std": float(np.std(bal_acc_scores)),
            "scores": bal_acc_scores.tolist(),
        },
        "f1_macro": {
            "mean": float(np.mean(f1_scores)),
            "std": float(np.std(f1_scores)),
            "scores": f1_scores.tolist(),
        },
        "cv_metrics": cv_metrics,
    }


# ---------------------------------------------------------------------------
# Subsample experiments (learning curves)
# ---------------------------------------------------------------------------

def subsample_experiment(
    model_factory: callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fractions: list[float],
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Train on subsamples of training data and evaluate on test set."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    
    results = []
    np.random.seed(random_state)
    n_train = len(X_train)
    indices = np.arange(n_train)
    
    for frac in fractions:
        n_sub = int(n_train * frac)
        if n_sub < 2:
            continue
        if n_sub >= n_train:
            # Use full dataset
            X_sub = X_train
            y_sub = y_train
        else:
            # Stratified subsample
            sub_indices, _ = train_test_split(
                indices, train_size=n_sub, stratify=y_train, random_state=random_state
            )
            
            X_sub = X_train[sub_indices]
            y_sub = y_train[sub_indices]
        
        model = model_factory()
        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass
        elif hasattr(model, "decision_function"):
            try:
                from scipy.special import expit
                decision = model.decision_function(X_test)
                if decision.ndim == 1:
                    y_proba = np.column_stack([1 - expit(decision), expit(decision)])
                else:
                    y_proba = expit(decision)
            except Exception:
                pass
        
        metrics = compute_metrics(y_test, y_pred, y_proba)
        results.append({
            "fraction": frac,
            "n_samples": n_sub,
            "metrics": metrics,
        })
        print(f"  Subsample {frac:.0%} (n={n_sub}): acc={metrics['accuracy']:.4f}, bal_acc={metrics['balanced_accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")
    
    return results


# ---------------------------------------------------------------------------
# Main training/evaluation pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train_rows: list[dict],
    test_rows: list[dict],
    eval_rows: list[dict],
    protos_path: Path,
    model_names: list[str],
    embedding_model: str,
    embed_url: Optional[str],
    use_embeddings: bool,
    pca_dim: Optional[int],
    cv_folds: int,
    output_dir: Path,
    subsample_fractions: list[float],
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Full training and evaluation pipeline."""
    
    # Prepare labels
    y_train_amb, y_train_sub = prepare_labels(train_rows)
    y_test_amb, y_test_sub = prepare_labels(test_rows)
    y_eval_amb, y_eval_sub = prepare_labels(eval_rows)
    
    # Extract features
    print("\n=== Feature Extraction ===")
    if use_embeddings:
        print("Using raw embeddings...")
        X_train = extract_raw_embeddings(train_rows, embedding_model, embed_url)
        X_test = extract_raw_embeddings(test_rows, embedding_model, embed_url)
        X_eval = extract_raw_embeddings(eval_rows, embedding_model, embed_url)
        feature_type = "raw_embeddings"
        
        # Apply PCA if requested
        if pca_dim:
            X_train, X_test, X_eval, pca = apply_pca(X_train, X_test, X_eval, pca_dim)
            feature_type += f"_pca{pca_dim}"
        else:
            pca = None
    else:
        print("Using 7-dim subspace features...")
        X_train = extract_features_7d(train_rows, protos_path, embedding_model, embed_url)
        X_test = extract_features_7d(test_rows, protos_path, embedding_model, embed_url)
        X_eval = extract_features_7d(eval_rows, protos_path, embedding_model, embed_url)
        feature_type = "subspace_7d"
        pca = None
    
    print(f"Feature shapes: train={X_train.shape}, test={X_test.shape}, eval={X_eval.shape}")
    
    # Get model instances
    models = get_models(model_names, random_state, n_jobs)
    if not models:
        raise ValueError("No valid models selected")
    
    # Unique labels for multiclass
    all_labels = sorted(set(y_train_sub.tolist()) | set(y_test_sub.tolist()) | set(y_eval_sub.tolist()))
    binary_labels = [0, 1]
    
    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_type": feature_type,
            "embedding_model": embedding_model,
            "train_size": len(train_rows),
            "test_size": len(test_rows),
            "eval_size": len(eval_rows),
            "n_features": int(X_train.shape[1]),
            "models": model_names,
            "cv_folds": cv_folds,
            "subsample_fractions": subsample_fractions,
            "random_state": random_state,
        },
        "models": {},
    }
    
    best_model_name = None
    best_model_obj = None
    best_bal_acc = -1.0
    
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        # Clone model for fresh training
        from sklearn.base import clone
        model = clone(model)
        
        # Cross-validation on train set (multiclass)
        print(f"  Cross-validation ({cv_folds}-fold)...")
        cv_results = cross_validate_model(model, X_train, y_train_sub, cv_folds, random_state)
        print(f"    CV Balanced Acc: {cv_results['balanced_accuracy']['mean']:.4f} ± {cv_results['balanced_accuracy']['std']:.4f}")
        print(f"    CV F1 Macro:     {cv_results['f1_macro']['mean']:.4f} ± {cv_results['f1_macro']['std']:.4f}")
        
        # Train on full train set
        model.fit(X_train, y_train_sub)
        
        # Test set evaluation (multiclass)
        print("  Test set evaluation...")
        y_test_pred = model.predict(X_test)
        y_test_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_test_proba = model.predict_proba(X_test)
            except Exception:
                pass
        elif hasattr(model, "decision_function"):
            try:
                from scipy.special import expit
                decision = model.decision_function(X_test)
                if decision.ndim == 1:
                    y_test_proba = np.column_stack([1 - expit(decision), expit(decision)])
                else:
                    y_test_proba = expit(decision)
            except Exception:
                pass
        
        test_metrics = compute_metrics(y_test_sub, y_test_pred, y_test_proba, all_labels)
        print(f"    Test Acc: {test_metrics['accuracy']:.4f}, Bal Acc: {test_metrics['balanced_accuracy']:.4f}, F1 Macro: {test_metrics['f1_macro']:.4f}")
        
        # Eval set evaluation (multiclass)
        print("  Eval set evaluation...")
        y_eval_pred = model.predict(X_eval)
        y_eval_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_eval_proba = model.predict_proba(X_eval)
            except Exception:
                pass
        elif hasattr(model, "decision_function"):
            try:
                from scipy.special import expit
                decision = model.decision_function(X_eval)
                if decision.ndim == 1:
                    y_eval_proba = np.column_stack([1 - expit(decision), expit(decision)])
                else:
                    y_eval_proba = expit(decision)
            except Exception:
                pass
        
        eval_metrics = compute_metrics(y_eval_sub, y_eval_pred, y_eval_proba, all_labels)
        print(f"    Eval Acc: {eval_metrics['accuracy']:.4f}, Bal Acc: {eval_metrics['balanced_accuracy']:.4f}, F1 Macro: {eval_metrics['f1_macro']:.4f}")
        
        # Binary classification (ambiguous vs non-ambiguous)
        print("  Binary (ambiguous vs non-ambiguous) evaluation...")
        y_test_amb_pred = (y_test_pred == "ambiguous").astype(int)
        y_eval_amb_pred = (y_eval_pred == "ambiguous").astype(int)
        
        test_binary = compute_binary_metrics(y_test_amb, y_test_amb_pred, y_test_proba)
        eval_binary = compute_binary_metrics(y_eval_amb, y_eval_amb_pred, y_eval_proba)
        print(f"    Test Binary: Acc={test_binary['accuracy']:.4f}, Bal Acc={test_binary['balanced_accuracy']:.4f}, F1={test_binary['f1']:.4f}")
        print(f"    Eval Binary: Acc={eval_binary['accuracy']:.4f}, Bal Acc={eval_binary['balanced_accuracy']:.4f}, F1={eval_binary['f1']:.4f}")
        
        # Subsample experiments
        print(f"  Subsample experiments...")
        def model_factory():
            return clone(model)
        
        subsample_results = subsample_experiment(
            model_factory, X_train, y_train_sub, X_test, y_test_sub,
            subsample_fractions, random_state
        )
        
        # Store results
        model_result = {
            "cv": cv_results,
            "test": test_metrics,
            "eval": eval_metrics,
            "test_binary": test_binary,
            "eval_binary": eval_binary,
            "subsample": subsample_results,
            "predictions": {
                "test": {
                    "y_true": y_test_sub.tolist(),
                    "y_pred": y_test_pred.tolist(),
                },
                "eval": {
                    "y_true": y_eval_sub.tolist(),
                    "y_pred": y_eval_pred.tolist(),
                },
            },
        }
        
        results["models"][model_name] = model_result
        
        # Track best model by eval balanced accuracy
        eval_bal_acc = eval_metrics["balanced_accuracy"]
        if eval_bal_acc > best_bal_acc:
            best_bal_acc = eval_bal_acc
            best_model_name = model_name
            best_model_obj = model
    
    # Save best model
    if best_model_obj is not None:
        import joblib
        best_model_path = output_dir / "best_ambiguity_model.joblib"
        joblib.dump({
            "model": best_model_obj,
            "model_name": best_model_name,
            "feature_type": feature_type,
            "feature_names": FEATURE_NAMES_7D if not use_embeddings else [f"emb_{i}" for i in range(X_train.shape[1])],
            "embedding_model": embedding_model,
            "pca": pca,
            "classes": all_labels,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "eval_balanced_accuracy": best_bal_acc,
        }, best_model_path)
        print(f"\nBest model: {best_model_name} (eval balanced acc = {best_bal_acc:.4f})")
        print(f"Saved to: {best_model_path}")
        results["best_model"] = {
            "name": best_model_name,
            "eval_balanced_accuracy": best_bal_acc,
            "path": str(best_model_path),
        }
    
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_json_report(results: dict, output_dir: Path) -> Path:
    """Save results as JSON."""
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    return json_path


def generate_markdown_report(results: dict, output_dir: Path) -> Path:
    """Generate comprehensive markdown report."""
    md_path = output_dir / "evaluation_report.md"
    
    meta = results["metadata"]
    models = results["models"]
    best = results.get("best_model", {})
    
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Ambiguity Classifier Evaluation Report\n\n")
        fh.write(f"**Generated:** {meta['timestamp']}\n\n")
        
        # Metadata
        fh.write("## Experiment Configuration\n\n")
        fh.write(f"- **Feature type:** {meta['feature_type']}\n")
        fh.write(f"- **Embedding model:** {meta['embedding_model']}\n")
        fh.write(f"- **Train size:** {meta['train_size']}\n")
        fh.write(f"- **Test size:** {meta['test_size']}\n")
        fh.write(f"- **Eval size:** {meta['eval_size']}\n")
        fh.write(f"- **Number of features:** {meta['n_features']}\n")
        fh.write(f"- **Models evaluated:** {', '.join(meta['models'])}\n")
        fh.write(f"- **CV folds:** {meta['cv_folds']}\n")
        fh.write(f"- **Subsample fractions:** {meta['subsample_fractions']}\n")
        fh.write(f"- **Random state:** {meta['random_state']}\n\n")
        
        # Best model summary
        fh.write("## Best Model\n\n")
        if best:
            fh.write(f"- **Model:** {best['name']}\n")
            fh.write(f"- **Eval Balanced Accuracy:** {best['eval_balanced_accuracy']:.4f}\n")
            fh.write(f"- **Saved to:** {best['path']}\n\n")
        
        # Model comparison table
        fh.write("## Model Comparison (Multiclass - Eval Set)\n\n")
        fh.write("| Model | Accuracy | Balanced Acc | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR) |\n")
        fh.write("|-------|----------|--------------|-------------------|----------------|------------|---------------|\n")
        for name, res in models.items():
            eval_m = res["eval"]
            roc_auc = eval_m.get("roc_auc_ovr", eval_m.get("roc_auc", "N/A"))
            fh.write(f"| {name} | {eval_m['accuracy']:.4f} | {eval_m['balanced_accuracy']:.4f} | "
                     f"{eval_m['precision_macro']:.4f} | {eval_m['recall_macro']:.4f} | "
                     f"{eval_m['f1_macro']:.4f} | {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'} |\n")
        fh.write("\n")
        
        # Binary comparison
        fh.write("## Binary Classification (Ambiguous vs Non-Ambiguous - Eval Set)\n\n")
        fh.write("| Model | Accuracy | Balanced Acc | Precision | Recall | F1 | ROC-AUC |\n")
        fh.write("|-------|----------|--------------|-----------|--------|-----|--------|\n")
        for name, res in models.items():
            bm = res["eval_binary"]
            roc = bm.get("roc_auc", "N/A")
            fh.write(f"| {name} | {bm['accuracy']:.4f} | {bm['balanced_accuracy']:.4f} | "
                     f"{bm['precision']:.4f} | {bm['recall']:.4f} | {bm['f1']:.4f} | {roc if isinstance(roc, str) else f'{roc:.4f}'} |\n")
        fh.write("\n")
        
        # Cross-validation results
        fh.write("## Cross-Validation Results (Train Set)\n\n")
        fh.write("| Model | CV Acc (mean±std) | CV Bal Acc (mean±std) | CV F1 Macro (mean±std) |\n")
        fh.write("|-------|-------------------|----------------------|------------------------|\n")
        for name, res in models.items():
            cv = res["cv"]
            fh.write(f"| {name} | {cv['accuracy']['mean']:.4f}±{cv['accuracy']['std']:.4f} | "
                     f"{cv['balanced_accuracy']['mean']:.4f}±{cv['balanced_accuracy']['std']:.4f} | "
                     f"{cv['f1_macro']['mean']:.4f}±{cv['f1_macro']['std']:.4f} |\n")
        fh.write("\n")
        
        # Per-class metrics for best model
        if best and best["name"] in models:
            best_res = models[best["name"]]["eval"]
            fh.write(f"## Per-Class Metrics (Best Model: {best['name']} - Eval Set)\n\n")
            fh.write("| Class | Precision | Recall | F1 | Support |\n")
            fh.write("|-------|-----------|--------|-----|--------|\n")
            for cls, metrics in best_res["per_class"].items():
                fh.write(f"| {cls} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |\n")
            fh.write("\n")
            
            # Confusion matrix
            fh.write("### Confusion Matrix (Eval Set)\n\n")
            cm = best_res["confusion_matrix"]
            labels = best_res["confusion_matrix_labels"]
            fh.write("| | " + " | ".join(labels) + " |\n")
            fh.write("|" + "---|" * (len(labels) + 1) + "\n")
            for i, row_label in enumerate(labels):
                row = [str(cm[i][j]) for j in range(len(labels))]
                fh.write(f"| {row_label} | " + " | ".join(row) + " |\n")
            fh.write("\n")
        
        # Learning curves
        fh.write("## Learning Curves (Subsample Experiments)\n\n")
        for name, res in models.items():
            fh.write(f"### {name}\n\n")
            fh.write("| Fraction | Train Samples | Test Acc | Test Bal Acc | Test F1 Macro |\n")
            fh.write("|----------|---------------|----------|--------------|---------------|\n")
            for sub in res["subsample"]:
                m = sub["metrics"]
                fh.write(f"| {sub['fraction']:.0%} | {sub['n_samples']} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['f1_macro']:.4f} |\n")
            fh.write("\n")
        
        # Key findings
        fh.write("## Key Findings\n\n")
        
        # Rank models by eval balanced accuracy
        ranked = sorted(models.items(), key=lambda x: x[1]["eval"]["balanced_accuracy"], reverse=True)
        fh.write("1. **Model Ranking (by Eval Balanced Accuracy):**\n")
        for i, (name, res) in enumerate(ranked, 1):
            fh.write(f"   {i}. {name}: {res['eval']['balanced_accuracy']:.4f}\n")
        fh.write("\n")
        
        # Best binary classifier
        ranked_binary = sorted(models.items(), key=lambda x: x[1]["eval_binary"]["balanced_accuracy"], reverse=True)
        fh.write("2. **Binary Classification Ranking (Ambiguous vs Non-Ambiguous):**\n")
        for i, (name, res) in enumerate(ranked_binary, 1):
            fh.write(f"   {i}. {name}: {res['eval_binary']['balanced_accuracy']:.4f}\n")
        fh.write("\n")
        
        # Overfitting check
        fh.write("3. **Overfitting Check (Train CV vs Eval Balanced Acc):**\n")
        for name, res in models.items():
            cv_bal = res["cv"]["balanced_accuracy"]["mean"]
            eval_bal = res["eval"]["balanced_accuracy"]
            diff = cv_bal - eval_bal
            fh.write(f"   - {name}: CV={cv_bal:.4f}, Eval={eval_bal:.4f}, Diff={diff:+.4f} {'⚠️' if diff > 0.05 else ''}\n")
        fh.write("\n")
        
        # Feature type comparison note
        fh.write("4. **Feature Type:** ")
        if meta["feature_type"].startswith("subspace"):
            fh.write("7-dimensional subspace features (4 cosine sims + margin + top1 + entropy). ")
            fh.write("These are interpretable and computationally efficient.\n\n")
        else:
            fh.write(f"{meta['n_features']}-dimensional raw embeddings")
            if "pca" in meta["feature_type"]:
                fh.write(f" with PCA({meta['feature_type'].split('pca')[1]})")
            fh.write(". Higher capacity but more prone to overfitting.\n\n")
    
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate ambiguity classifiers")
    parser.add_argument("--train", type=Path, required=True, help="Training data JSONL")
    parser.add_argument("--test", type=Path, required=True, help="Test data JSONL")
    parser.add_argument("--eval", type=Path, required=True, help="Eval data JSONL")
    parser.add_argument("--protos", type=Path, default="data/subspace_protos.npz", help="Subspace prototypes NPZ")
    parser.add_argument("--models", default="lr,svm_linear,svm_rbf,rf,mlp,xgb", help="Comma-separated model names")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="Embedding model id")
    parser.add_argument("--embed-url", default=None, help="Embedding service base URL")
    parser.add_argument("--use-embeddings", action="store_true", help="Use raw embeddings instead of 7-dim features")
    parser.add_argument("--pca-dim", type=int, default=None, help="PCA dimensions for raw embeddings (default: no PCA)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--output-dir", type=Path, default="data/model_eval/", help="Output directory")
    parser.add_argument("--subsample-fractions", default="0.25,0.5,0.75,1.0", help="Comma-separated subsample fractions")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (-1 for all)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Resolve paths relative to orchestrator root
    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else ORCH_ROOT / p
    
    train_path = resolve(args.train)
    test_path = resolve(args.test)
    eval_path = resolve(args.eval)
    protos_path = resolve(args.protos)
    output_dir = resolve(args.output_dir)
    
    # Verify files exist
    for p, name in [(train_path, "train"), (test_path, "test"), (eval_path, "eval"), (protos_path, "protos")]:
        if not p.exists():
            print(f"[error] {name} file not found: {p}")
            return 1
    
    # Parse model names and subsample fractions
    model_names = [m.strip() for m in args.models.split(",")]
    subsample_fractions = [float(f.strip()) for f in args.subsample_fractions.split(",")]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Ambiguity Classifier Training & Evaluation")
    print("=" * 60)
    
    # Load data
    train_rows, test_rows, eval_rows = load_splits(train_path, test_path, eval_path)
    
    # Run pipeline
    results = train_and_evaluate(
        train_rows, test_rows, eval_rows,
        protos_path, model_names,
        args.embedding_model, args.embed_url,
        args.use_embeddings, args.pca_dim,
        args.cv_folds, output_dir,
        subsample_fractions,
        args.random_state, args.n_jobs,
    )
    
    # Generate reports
    json_path = generate_json_report(results, output_dir)
    md_path = generate_markdown_report(results, output_dir)
    
    print(f"\n=== Reports Generated ===")
    print(f"JSON:  {json_path}")
    print(f"MD:    {md_path}")
    
    # Print summary
    best = results.get("best_model", {})
    if best:
        print(f"\nBest model: {best['name']} (eval balanced accuracy = {best['eval_balanced_accuracy']:.4f})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())