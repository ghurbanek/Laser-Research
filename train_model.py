"""Train an aberration-type classifier and one strength regressor per type.

Install:
    python -m pip install numpy pillow scikit-learn joblib

Place this file beside training_data/. Expected inputs:
    training_data/training_coma/coma_0.5.png
    training_data/training_defocus/defocus_0.5.png

Run:
    python train_model.py
    python train_model.py predict new_beam.png
    python train_model.py train --color-mode L

Use --color-mode L for PNGs generated directly with a grayscale mapping.
RGB is the default to support your existing colored PNG dataset. Do not mix
different colormaps, crops, or physical fields of view within a model.

This first version uses positive coefficients, skips zero-strength examples,
and expects exactly one aberration type per image. It estimates the numerical
coefficient used by your simulator; it does not infer its physical units.
"""

from pathlib import Path
from collections import Counter
import argparse
import hashlib

import numpy as np
import joblib
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error


SCRIPT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = SCRIPT_FOLDER / "training_data"
MODEL_PATH = SCRIPT_FOLDER / "aberration_strength_model.joblib"
IMAGE_SIZE = (64, 64)
COLOR_MODE = "RGB"
RANDOM_STATE = 42


def load_image(image_path, image_size=IMAGE_SIZE, color_mode=COLOR_MODE):
    """Load a PNG using the same preparation for training and prediction."""
    with Image.open(image_path) as image:
        image = image.convert(color_mode)
        image = image.resize(image_size, resample=Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.flatten()


def load_dataset(data_folder, image_size=IMAGE_SIZE, color_mode=COLOR_MODE):
    """Read folder labels and filename coefficients; identify duplicate inputs."""
    data_folder = Path(data_folder)
    if not data_folder.is_dir():
        raise ValueError(f"Dataset folder does not exist: {data_folder}")

    images, labels, strengths, groups = [], [], [], []
    group_labels = {}
    skipped_zero = 0

    for folder in sorted(data_folder.glob("training_*")):
        if not folder.is_dir():
            continue
        label = folder.name.removeprefix("training_")
        if not label:
            raise ValueError(f"Folder has no aberration name: {folder}")

        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() != ".png":
                continue
            try:
                strength = float(image_path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"Expected a filename such as coma_0.5.png: {image_path}"
                ) from error

            if not np.isfinite(strength):
                raise ValueError(f"Coefficient must be finite: {image_path}")
            if strength == 0:
                skipped_zero += 1
                continue
            if strength < 0:
                raise ValueError(
                    f"Negative coefficient in {image_path}. This version trains "
                    "on positive coefficients. Check whether your optical setup "
                    "can distinguish coefficient signs before using signed targets."
                )

            features = load_image(image_path, image_size, color_mode)
            group = hashlib.sha256(features.tobytes()).hexdigest()
            if group in group_labels and group_labels[group] != label:
                raise ValueError(
                    f"Identical processed images have conflicting type labels "
                    f"{group_labels[group]!r} and {label!r}; found at {image_path}. "
                    "Check labels, image resolution, and observable aberrations."
                )
            group_labels[group] = label
            images.append(features)
            labels.append(label)
            strengths.append(strength)
            groups.append(group)

    if not images:
        raise ValueError(
            f"No positive-strength PNGs found in {data_folder}/training_*/."
        )
    counts = Counter(labels)
    unique_counts = Counter(group_labels.values())
    print("Images per class:", dict(counts))
    print("Distinct processed images per class:", dict(unique_counts))
    print("Zero-strength images skipped:", skipped_zero)

    if len(counts) < 2:
        raise ValueError("You need at least two aberration types.")
    if min(unique_counts.values()) < 5:
        raise ValueError("Use at least five distinct processed images per type.")

    return (
        np.stack(images), np.asarray(labels),
        np.asarray(strengths, dtype=np.float64), np.asarray(groups), group_labels
    )


def train_model(data_folder=DATA_FOLDER, model_path=MODEL_PATH,
                image_size=IMAGE_SIZE, color_mode=COLOR_MODE):
    X, types, strengths, groups, group_labels = load_dataset(
        data_folder, image_size, color_mode
    )

    # Split unique processed images, keeping identical copies in the same set.
    # Stratification maintains representation of every aberration type.
    unique_groups = np.asarray(list(group_labels))
    unique_types = np.asarray([group_labels[group] for group in unique_groups])
    train_groups, test_groups = train_test_split(
        unique_groups, test_size=0.20,
        stratify=unique_types, random_state=RANDOM_STATE
    )
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    X_train, X_test = X[train_mask], X[test_mask]
    type_train, type_test = types[train_mask], types[test_mask]
    strength_train, strength_test = strengths[train_mask], strengths[test_mask]

    print("Input shape:", X.shape)
    print(f"Training images: {len(X_train)} | Test images: {len(X_test)}")

    classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
    classifier.fit(X_train, type_train)
    type_predictions = classifier.predict(X_test)
    type_accuracy = float(accuracy_score(type_test, type_predictions))
    print(f"Type accuracy: {type_accuracy:.1%}")
    print(classification_report(type_test, type_predictions, zero_division=0))

    regressors, strength_ranges, known_type_errors = {}, {}, {}
    routed_predictions = np.empty(len(X_test), dtype=np.float64)

    for label in classifier.classes_:
        training = type_train == label
        testing = type_test == label
        targets = strength_train[training]
        if np.unique(targets).size < 2:
            raise ValueError(f"{label}: need multiple strengths in the training set.")

        regressor = make_pipeline(
            StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.01)
        )
        regressor.fit(X_train[training], targets)
        predictions = regressor.predict(X_test[testing])
        error = float(mean_absolute_error(strength_test[testing], predictions))
        print(f"{label}: strength MAE with known type = {error:.4f}")

        predicted_as_type = type_predictions == label
        if predicted_as_type.any():
            routed_predictions[predicted_as_type] = regressor.predict(
                X_test[predicted_as_type]
            )
        key = str(label)
        regressors[key] = regressor
        strength_ranges[key] = (float(targets.min()), float(targets.max()))
        known_type_errors[key] = error

    correct_type = type_predictions == type_test
    conditional_error = None
    if correct_type.any():
        conditional_error = float(mean_absolute_error(
            strength_test[correct_type], routed_predictions[correct_type]
        ))
        print(
            "Strength MAE on correctly classified test images: "
            f"{conditional_error:.4f} "
            f"({correct_type.sum()}/{len(type_test)} images)"
        )

    artifact = {
        "classifier": classifier, "regressors": regressors,
        "image_size": tuple(image_size), "color_mode": color_mode,
        "strength_ranges": strength_ranges, "target": "positive simulator coefficient",
        "metrics": {
            "type_accuracy": type_accuracy,
            "known_type_strength_mae": known_type_errors,
            "correct_type_strength_mae": conditional_error,
            "train_images": len(X_train), "test_images": len(X_test)
        },
        "sklearn_version": __import__("sklearn").__version__
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"Saved model to: {model_path.resolve()}")
    return artifact


def predict_aberration(image_path, model_path=MODEL_PATH):
    """Predict one type and its coefficient, using the saved image preparation."""
    # Only load joblib model files from a trusted source.
    artifact = joblib.load(model_path)
    features = load_image(
        image_path, tuple(artifact["image_size"]), artifact["color_mode"]
    )[None, :]
    predicted_type = str(artifact["classifier"].predict(features)[0])
    predicted_strength = float(
        artifact["regressors"][predicted_type].predict(features)[0]
    )
    return predicted_type, predicted_strength


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", nargs="?", choices=("train", "predict"), default="train")
    parser.add_argument("image", nargs="?", help="Image to classify when action is predict")
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--color-mode", choices=("RGB", "L"), default=COLOR_MODE,
                        help="Training mode; prediction uses the mode saved in the model")
    args = parser.parse_args()
    try:
        if args.action == "train":
            if args.image:
                parser.error("Use 'predict image.png' to classify an image.")
            train_model(args.data_folder, args.model, color_mode=args.color_mode)
        else:
            if not args.image:
                parser.error("Provide an image: python train_model.py predict image.png")
            kind, strength = predict_aberration(args.image, args.model)
            print("Estimated aberration:", kind)
            print(f"Estimated coefficient: {strength:.4f}")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
    print("Done")