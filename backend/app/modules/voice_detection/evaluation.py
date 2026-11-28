import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_voice_model(model, X_test, y_test):
    """
    Evaluate the trained voice detection model.

    Label 0 = Genuine speech
    Label 1 = AI-generated speech
    """

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test features and labels have different lengths."
        )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    print("\nVOICE DETECTION MODEL EVALUATION")
    print("=" * 40)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Genuine Speech",
                "AI-Generated Speech"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def evaluate_language_wise(
    model,
    X_test,
    y_test,
    languages
):
    """
    Evaluate voice detection performance separately
    for each language.
    """

    if not (
        len(X_test) == len(y_test) == len(languages)
    ):
        raise ValueError(
            "Test features, labels, and languages "
            "must have equal lengths."
        )

    predictions = model.predict(X_test)

    results = []

    for language in sorted(set(languages)):

        indices = [
            i for i, lang in enumerate(languages)
            if lang == language
        ]

        y_true = y_test[indices]
        y_pred = predictions[indices]

        results.append({
            "language": language,
            "samples": len(indices),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    print("\nLANGUAGE-WISE EVALUATION")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return results_df


def evaluate_language_and_class_wise(
    model,
    X_test,
    y_test,
    languages
):
    """
    Evaluate genuine and AI-generated speech separately
    for every language.

    Label 0 = Genuine speech
    Label 1 = AI-generated speech
    """

    if not (
        len(X_test) == len(y_test) == len(languages)
    ):
        raise ValueError(
            "Features, labels, and languages must have equal lengths."
        )

    predictions = model.predict(X_test)

    results = []

    for language in sorted(set(languages)):

        language_indices = np.where(
            np.asarray(languages) == language
        )[0]

        true_labels = np.asarray(y_test)[language_indices]
        predicted_labels = predictions[language_indices]

        for label, class_name in [
            (0, "Genuine"),
            (1, "AI-generated")
        ]:

            class_indices = np.where(
                true_labels == label
            )[0]

            total = len(class_indices)

            if total == 0:
                continue

            correct = np.sum(
                predicted_labels[class_indices] == label
            )

            results.append({
                "language": language,
                "class": class_name,
                "total_samples": int(total),
                "correct": int(correct),
                "incorrect": int(total - correct),
                "recall_percent": round(
                    float(correct / total * 100),
                    2
                )
            })

    results_df = pd.DataFrame(results)

    print("\nLANGUAGE AND CLASS-WISE EVALUATION")
    print("=" * 75)

    if results_df.empty:
        print("No evaluation results available.")
    else:
        print(results_df.to_string(index=False))

    return results_df