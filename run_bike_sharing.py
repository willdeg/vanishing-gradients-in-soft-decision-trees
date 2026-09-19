from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add the repo root so the script can be run directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baseline import LogisticRegressionGD
from scripts.data import load_bike_sharing_binary, standardize_from_train, train_val_test_split
from scripts.metrics import classification_metrics, metrics_to_frame
from scripts.soft_tree import initialize_root, learn_soft_tree, plot_gradient_boxplot, predict, prediction_loss


SEED = 42
TARGET_QUANTILE = 0.75
LEARNING_RATE = 0.01
MAX_ITER = 10
MAX_DEPTH = 6
LOGISTIC_LEARNING_RATE = 0.1
LOGISTIC_MAX_ITER = 1000
LOGISTIC_TOL = 1e-4
SAVE_PLOTS = True
OUTPUT_DIR = REPO_ROOT / "outputs"


def run_logistic_baseline(train_x, val_x, test_x, train_y, val_y, test_y):
    train_x_std, [val_x_std, test_x_std], _, _ = standardize_from_train(train_x, [val_x, test_x])

    model = LogisticRegressionGD()
    model.fit(
        train_x_std,
        train_y,
        learning_rate=LOGISTIC_LEARNING_RATE,
        n_iter=LOGISTIC_MAX_ITER,
        tol=LOGISTIC_TOL,
    )

    val_prob = model.predict_proba(val_x_std)
    val_pred = model.predict(val_x_std)
    test_prob = model.predict_proba(test_x_std)
    test_pred = model.predict(test_x_std)

    return (
        classification_metrics(val_y, val_pred, val_prob),
        classification_metrics(test_y, test_pred, test_prob),
    )


def run_soft_tree_model(train_x, val_x, test_x, train_y, val_y, test_y, use_depth_scaling):
    # The only difference between ordinary and scaled SDT should be this flg
    root = initialize_root(train_x.shape[1], train_y.mean())

    grad_recorder = learn_soft_tree(
        node=root,
        x_train=train_x,
        y_train=train_y,
        x_val=val_x,
        y_val=val_y,
        root=root,
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        max_depth=MAX_DEPTH,
        use_depth_scaling=use_depth_scaling,
        grad_recorder={},
        verbose=False,
    )

    val_pred, val_prob = predict(root, val_x)
    test_pred, test_prob = predict(root, test_x)

    val_metrics = classification_metrics(val_y, val_pred, val_prob)
    test_metrics = classification_metrics(test_y, test_pred, test_prob)
    val_loss = prediction_loss(root, val_x, val_y, task="classification")
    test_loss = prediction_loss(root, test_x, test_y, task="classification")

    return val_metrics, test_metrics, grad_recorder, val_loss, test_loss


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # same split
    x, y = load_bike_sharing_binary(target_quantile=TARGET_QUANTILE)
    split = train_val_test_split(x, y, random_state=SEED)

    # standardized features for both trees
    train_x_std, [val_x_std, test_x_std], _, _ = standardize_from_train(
        split.train_x,
        [split.val_x, split.test_x],
    )

    baseline_val, baseline_test = run_logistic_baseline(
        split.train_x,
        split.val_x,
        split.test_x,
        split.train_y,
        split.val_y,
        split.test_y,
    )

    ordinary_val, ordinary_test, ordinary_grads, ordinary_val_loss, ordinary_test_loss = run_soft_tree_model(
        train_x_std,
        val_x_std,
        test_x_std,
        split.train_y,
        split.val_y,
        split.test_y,
        use_depth_scaling=False,
    )

    scaled_val, scaled_test, scaled_grads, scaled_val_loss, scaled_test_loss = run_soft_tree_model(
        train_x_std,
        val_x_std,
        test_x_std,
        split.train_y,
        split.val_y,
        split.test_y,
        use_depth_scaling=True,
    )

    summary = metrics_to_frame(
        [
            ("Baseline Logistic Regression", baseline_test),
            ("Ordinary Soft Decision Tree", ordinary_test),
            ("Scaled Soft Decision Tree", scaled_test),
        ]
    )

    validation_summary = metrics_to_frame(
        [
            ("Baseline Logistic Regression", baseline_val),
            ("Ordinary Soft Decision Tree", ordinary_val),
            ("Scaled Soft Decision Tree", scaled_val),
        ]
    )

    loss_summary = pd.DataFrame(
        [
            {
                "Model": "Ordinary Soft Decision Tree",
                "ValidationLogLoss": ordinary_val_loss,
                "TestLogLoss": ordinary_test_loss,
            },
            {
                "Model": "Scaled Soft Decision Tree",
                "ValidationLogLoss": scaled_val_loss,
                "TestLogLoss": scaled_test_loss,
            },
        ]
    )

    summary.to_csv(OUTPUT_DIR / "bike_sharing_summary.csv", index=False)
    validation_summary.to_csv(OUTPUT_DIR / "bike_sharing_validation_metrics.csv", index=False)
    loss_summary.to_csv(OUTPUT_DIR / "soft_tree_losses.csv", index=False)

    if SAVE_PLOTS:
        plot_gradient_boxplot(ordinary_grads, save_path=str(OUTPUT_DIR / "ordinary_soft_tree_gradients.png"))
        plot_gradient_boxplot(scaled_grads, save_path=str(OUTPUT_DIR / "scaled_soft_tree_gradients.png"))

    print("\nValidation summary")
    print(validation_summary.to_string(index=False))
    print("\nTest summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
