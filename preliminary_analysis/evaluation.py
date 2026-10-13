"""Reusable subject-only folds and training-only preprocessing for every representation."""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["AD", "FTD", "HC"]


def saved_folds(subjects, output, seed):
    if not subjects.participant_id.is_unique:
        raise ValueError("Evaluation requires exactly one row per participant")
    if subjects.group.value_counts().min() < 5 or set(subjects.group) != set(CLASSES):
        raise ValueError("Five folds require at least five usable participants in each class")
    path = Path(output) / "fold_assignments.csv"
    expected = subjects[["participant_id", "group"]].sort_values("participant_id").reset_index(drop=True)
    expected["fold"] = -1
    for fold, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(expected, expected.group)):
        expected.loc[test, "fold"] = fold
    if path.exists():
        actual = pd.read_csv(path).sort_values("participant_id").reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError("Existing subject folds differ: use a new output directory")
    else:
        expected.to_csv(path, index=False)
    return expected


def score(y, predicted):
    return {"balanced_accuracy": balanced_accuracy_score(y, predicted),
            "macro_f1": f1_score(y, predicted, labels=CLASSES, average="macro", zero_division=0),
            **dict(zip([f"recall_{c}" for c in CLASSES],
                       recall_score(y, predicted, labels=CLASSES, average=None, zero_division=0)))}


def evaluate(subjects, columns, folds, output, config, representation="spectral", include_dummy=True):
    output = Path(output)
    merged = subjects.merge(folds, on=["participant_id", "group"], validate="one_to_one").sort_values("participant_id")
    if len(merged) != len(subjects):
        raise ValueError("Incomplete fold coverage")
    x = merged[columns].to_numpy(dtype=float)
    y = merged.group.to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Participant features must be finite before evaluation")
    classifiers = ["logistic"] + (["dummy"] if include_dummy else [])
    rows, predictions, checks = [], [], []
    for name in classifiers:
        covered = np.zeros(len(y), dtype=int)
        for fold in range(5):
            train, test = np.flatnonzero(merged.fold.to_numpy() != fold), np.flatnonzero(merged.fold.to_numpy() == fold)
            train_ids, test_ids = set(merged.iloc[train].participant_id), set(merged.iloc[test].participant_id)
            if train_ids & test_ids:
                raise ValueError("Participant leakage")
            if name == "logistic":
                # lbfgs uses the joint multinomial objective for three classes.
                model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scaler", StandardScaler()),
                                  ("classifier", LogisticRegression(C=config["logistic_C"], solver="lbfgs",
                                                                  class_weight=config["class_weight"],
                                                                  max_iter=5000, tol=1e-6,
                                                                  random_state=config["seed"]))])
            else:
                model = DummyClassifier(strategy="prior", random_state=config["seed"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x[train], y[train])
            predicted, probability = model.predict(x[test]), model.predict_proba(x[test])
            if name == "logistic":
                transformed_train = model["imputer"].transform(x[train])
                np.testing.assert_allclose(model["scaler"].mean_, transformed_train.mean(axis=0), atol=1e-12)
                assert np.all(model["scaler"].n_samples_seen_ == len(train))
                np.testing.assert_allclose(model["imputer"].statistics_, np.median(x[train], axis=0), atol=1e-12)
            np.testing.assert_allclose(probability.sum(axis=1), 1., atol=1e-12)
            rows.append({"representation": representation, "classifier": name, "fold": fold,
                         "n_train": len(train), "n_test": len(test), **score(y[test], predicted)})
            for i, index in enumerate(test):
                predictions.append({"participant_id": merged.iloc[index].participant_id, "group": y[index],
                                    "fold": fold, "representation": representation, "classifier": name,
                                    "prediction": predicted[i],
                                    **{f"probability_{c}": probability[i, list(model.classes_).index(c)] for c in CLASSES}})
            covered[test] += 1
            checks.append({"classifier": name, "fold": fold, "participant_overlap": 0,
                           "training_only_scaling_checked": name == "logistic"})
        if not np.all(covered == 1):
            raise ValueError("OOF predictions do not cover every subject exactly once")
    scores, oof = pd.DataFrame(rows), pd.DataFrame(predictions).sort_values(["classifier", "participant_id"])
    scores.to_csv(output / f"{representation}_fold_scores.csv", index=False)
    oof.to_csv(output / f"{representation}_oof_predictions.csv", index=False)
    summaries = []
    matrices = {}
    for name in classifiers:
        part = scores.loc[scores.classifier == name]
        pred = oof.loc[oof.classifier == name]
        matrix = confusion_matrix(pred.group, pred.prediction, labels=CLASSES)
        matrices[name] = matrix.tolist()
        pd.DataFrame(matrix, index=pd.Index(CLASSES, name="true_group"), columns=CLASSES).to_csv(
            output / f"{representation}_{name}_confusion_matrix.csv")
        summary = {"representation": representation, "classifier": name,
                   "n_participants": len(y), "n_features": len(columns)}
        for metric in ["balanced_accuracy", "macro_f1", "recall_AD", "recall_FTD", "recall_HC"]:
            summary[f"{metric}_fold_mean"] = part[metric].mean()
            summary[f"{metric}_fold_sd"] = part[metric].std(ddof=1)
        summary.update({f"{k}_pooled_oof": v for k, v in score(pred.group, pred.prediction).items()})
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output / f"{representation}_metrics.csv", index=False)
    (output / f"{representation}_validation.json").write_text(json.dumps({
        "checks": checks, "complete_oof_coverage": True, "finite_features": True,
        "participant_count": len(y), "feature_count": len(columns), "class_order": CLASSES,
        "confusion_matrices": matrices, "chance_balanced_accuracy": 1 / 3,
        "fold_sd": "Sample SD (ddof=1): descriptive split variability, not a confidence interval",
        "hyperparameter_search": False}, indent=2) + "\n")
    return pd.DataFrame(summaries), oof
