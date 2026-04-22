"""
Fas 0: Data-audit
Tabulerar generatortyper (model, type) i datasetet.
Kör detta INNAN övriga experiment för att förstå datafördelningen.
"""
import os
from collections import Counter
from datasets import load_from_disk, concatenate_datasets


def create_dataset():
    DATA_DIR = os.path.expanduser("data/openfake_arrow_chunks_50_50")
    chunk_dirs = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if d.startswith("chunk_")
    ])
    dataset_list = [load_from_disk(d) for d in chunk_dirs]
    return concatenate_datasets(dataset_list)


def main():
    print("Laddar dataset...")
    ds = create_dataset()
    ds = ds.class_encode_column("label")

    label_names = ds.features["label"].names  # ["fake", "real"]
    total = len(ds)

    print(f"\nTotalt antal bilder: {total}")
    print(f"Label-namn: {label_names}")

    # --- 1. Fördelning per type (GAN / Diffusion / etc) ---
    print("\n" + "=" * 60)
    print("FÖRDELNING PER TYPE")
    print("=" * 60)
    type_counts = Counter(ds["type"])
    for t, count in type_counts.most_common():
        pct = 100 * count / total
        print(f"  {t:30s}  {count:6d}  ({pct:.1f}%)")

    # --- 2. Fördelning per model ---
    print("\n" + "=" * 60)
    print("FÖRDELNING PER MODEL")
    print("=" * 60)
    model_counts = Counter(ds["model"])
    for m, count in model_counts.most_common():
        pct = 100 * count / total
        print(f"  {m:40s}  {count:6d}  ({pct:.1f}%)")

    # --- 3. Korsreferens: type x label ---
    print("\n" + "=" * 60)
    print("KORSREFERENS: TYPE x LABEL")
    print("=" * 60)
    types = ds["type"]
    labels = ds["label"]

    type_label = {}
    for t, l in zip(types, labels):
        if t not in type_label:
            type_label[t] = Counter()
        type_label[t][label_names[l]] += 1

    print(f"  {'Type':30s}  {'Real':>8s}  {'Fake':>8s}  {'Total':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    for t in sorted(type_label.keys()):
        r = type_label[t].get("real", 0)
        f = type_label[t].get("fake", 0)
        print(f"  {t:30s}  {r:8d}  {f:8d}  {r+f:8d}")

    # --- 4. Korsreferens: model x label ---
    print("\n" + "=" * 60)
    print("KORSREFERENS: MODEL x LABEL")
    print("=" * 60)
    models = ds["model"]

    model_label = {}
    for m, l in zip(models, labels):
        if m not in model_label:
            model_label[m] = Counter()
        model_label[m][label_names[l]] += 1

    print(f"  {'Model':40s}  {'Real':>8s}  {'Fake':>8s}  {'Total':>8s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")
    for m in sorted(model_label.keys()):
        r = model_label[m].get("real", 0)
        f = model_label[m].get("fake", 0)
        print(f"  {m:40s}  {r:8d}  {f:8d}  {r+f:8d}")

    # --- 5. Samma analys för enbart test-split ---
    print("\n\n" + "=" * 60)
    print("SAMMA ANALYS FÖR TEST-SPLIT")
    print("=" * 60)

    if os.path.exists("openfake_test_split"):
        test_ds = load_from_disk("openfake_test_split")
        test_total = len(test_ds)
        print(f"Test-set storlek: {test_total}")

        test_types = test_ds["type"]
        test_labels = test_ds["label"]
        test_label_names = test_ds.features["label"].names

        test_type_label = {}
        for t, l in zip(test_types, test_labels):
            if t not in test_type_label:
                test_type_label[t] = Counter()
            test_type_label[t][test_label_names[l]] += 1

        print(f"\n  {'Type':30s}  {'Real':>8s}  {'Fake':>8s}  {'Total':>8s}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
        for t in sorted(test_type_label.keys()):
            r = test_type_label[t].get("real", 0)
            f = test_type_label[t].get("fake", 0)
            print(f"  {t:30s}  {r:8d}  {f:8d}  {r+f:8d}")

        # Model breakdown i test
        test_models = test_ds["model"]
        test_model_label = {}
        for m, l in zip(test_models, test_labels):
            if m not in test_model_label:
                test_model_label[m] = Counter()
            test_model_label[m][test_label_names[l]] += 1

        print(f"\n  {'Model':40s}  {'Real':>8s}  {'Fake':>8s}  {'Total':>8s}")
        print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")
        for m in sorted(test_model_label.keys()):
            r = test_model_label[m].get("real", 0)
            f = test_model_label[m].get("fake", 0)
            print(f"  {m:40s}  {r:8d}  {f:8d}  {r+f:8d}")
    else:
        print("  openfake_test_split hittades inte!")


if __name__ == "__main__":
    main()
