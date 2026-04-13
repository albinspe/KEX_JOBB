import io
import json
import os
import time

from PIL import Image as PILImage, ImageFile, UnidentifiedImageError
from datasets import Dataset, Image, load_dataset
from tqdm.auto import tqdm

# ===== CONFIG =====
DATASET_NAME = "ComplexDataLab/OpenFake"
SPLITS = ["train", "test"]
TARGET_REAL = 50_000
TARGET_FAKE = 50_000
TOTAL_TARGET = TARGET_REAL + TARGET_FAKE

CHUNK_SIZE = 1_000
OUT_ROOT = "openfake_arrow_chunks_50_50"
STATE_FILE = os.path.join(OUT_ROOT, "progress.json")

MIN_SIDE = 448  # övre gräns borttagen
CROP_SIZE = 448

SHUFFLE_BUFFER_SIZE = 10_000
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0
LOG_EVERY_KEPT = 200
# ==================

os.makedirs(OUT_ROOT, exist_ok=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
    else:
        s = {}
    return {
        "chunk_idx": int(s.get("chunk_idx", 0)),
        "scanned_total": int(s.get("scanned_total", 0)),
        "decode_errors": int(s.get("decode_errors", 0)),
        "process_errors": int(s.get("process_errors", 0)),
        "filtered_out": int(s.get("filtered_out", 0)),
        "real_kept": int(s.get("real_kept", 0)),
        "fake_kept": int(s.get("fake_kept", 0)),
    }


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def normalize_label(label):
    if not isinstance(label, str):
        return None
    low = label.strip().lower()
    if low in {"real", "fake"}:
        return low
    return None


def process_image_to_jpeg_bytes(img):
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if not (w >= MIN_SIDE and h >= MIN_SIDE):
        return None

    left = (w - CROP_SIZE) // 2
    top = (h - CROP_SIZE) // 2
    cropped = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    return buf.getvalue()


def flush_chunk(rows, chunk_idx):
    if not rows:
        return 0.0
    t0 = time.time()
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("image", Image())
    ds.save_to_disk(f"{OUT_ROOT}/chunk_{chunk_idx:04d}")
    return time.time() - t0


def main():
    state = load_state()

    rows = []
    chunk_idx = state["chunk_idx"]
    scanned_total = state["scanned_total"]
    decode_errors = state["decode_errors"]
    process_errors = state["process_errors"]
    filtered_out = state["filtered_out"]
    real_kept = state["real_kept"]
    fake_kept = state["fake_kept"]

    kept = real_kept + fake_kept
    pbar = tqdm(total=TOTAL_TARGET, initial=min(kept, TOTAL_TARGET))

    start_time = time.time()
    decode_time = 0.0
    process_time = 0.0
    save_time = 0.0
    last_log_kept = kept

    for split in SPLITS:
        if real_kept >= TARGET_REAL and fake_kept >= TARGET_FAKE:
            break

        ds = load_dataset(DATASET_NAME, split=split, streaming=True)
        ds = ds.cast_column("image", Image(decode=False))
        ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)

        for ex in ds:
            if real_kept >= TARGET_REAL and fake_kept >= TARGET_FAKE:
                break

            scanned_total += 1

            label = normalize_label(ex.get("label"))
            if label is None:
                filtered_out += 1
                continue

            if label == "real" and real_kept >= TARGET_REAL:
                filtered_out += 1
                continue
            if label == "fake" and fake_kept >= TARGET_FAKE:
                filtered_out += 1
                continue

            t0 = time.time()
            try:
                img_obj = ex.get("image")
                if img_obj is None:
                    filtered_out += 1
                    continue

                if img_obj.get("bytes") is not None:
                    pil_img = PILImage.open(io.BytesIO(img_obj["bytes"]))
                elif img_obj.get("path") is not None:
                    pil_img = PILImage.open(img_obj["path"])
                else:
                    filtered_out += 1
                    continue

                pil_img.load()
            except (UnidentifiedImageError, OSError, ValueError):
                decode_errors += 1
                continue
            finally:
                decode_time += time.time() - t0

            t1 = time.time()
            try:
                jpg_bytes = process_image_to_jpeg_bytes(pil_img)
            except (OSError, ValueError):
                process_errors += 1
                continue
            process_time += time.time() - t1

            if jpg_bytes is None:
                filtered_out += 1
                continue

            ex["image"] = {"bytes": jpg_bytes, "path": None}
            rows.append(ex)

            if label == "real":
                real_kept += 1
            else:
                fake_kept += 1

            kept = real_kept + fake_kept
            pbar.update(1)

            if len(rows) >= CHUNK_SIZE:
                save_time += flush_chunk(rows, chunk_idx)
                rows = []
                chunk_idx += 1
                state.update(
                    {
                        "chunk_idx": chunk_idx,
                        "scanned_total": scanned_total,
                        "decode_errors": decode_errors,
                        "process_errors": process_errors,
                        "filtered_out": filtered_out,
                        "real_kept": real_kept,
                        "fake_kept": fake_kept,
                    }
                )
                save_state(state)
                print(
                    f"[chunk] chunk_{chunk_idx-1:04d} kept={kept} real={real_kept} "
                    f"fake={fake_kept} scanned={scanned_total}"
                )

            if kept - last_log_kept >= LOG_EVERY_KEPT:
                elapsed = time.time() - start_time
                keep_rate = kept / elapsed if elapsed > 0 else 0.0
                scan_rate = scanned_total / elapsed if elapsed > 0 else 0.0
                print(
                    f"[perf] kept={kept} real={real_kept} fake={fake_kept} "
                    f"keep_rate={keep_rate:.2f}/s scan_rate={scan_rate:.2f}/s "
                    f"decode_err={decode_errors} process_err={process_errors} filtered={filtered_out}"
                )
                last_log_kept = kept
                state.update(
                    {
                        "chunk_idx": chunk_idx,
                        "scanned_total": scanned_total,
                        "decode_errors": decode_errors,
                        "process_errors": process_errors,
                        "filtered_out": filtered_out,
                        "real_kept": real_kept,
                        "fake_kept": fake_kept,
                    }
                )
                save_state(state)

    if rows:
        save_time += flush_chunk(rows, chunk_idx)
        chunk_idx += 1

    state.update(
        {
            "chunk_idx": chunk_idx,
            "scanned_total": scanned_total,
            "decode_errors": decode_errors,
            "process_errors": process_errors,
            "filtered_out": filtered_out,
            "real_kept": real_kept,
            "fake_kept": fake_kept,
        }
    )
    save_state(state)

    pbar.close()

    elapsed = time.time() - start_time
    print(f"Klart: kept={real_kept + fake_kept}, real={real_kept}, fake={fake_kept}, out={OUT_ROOT}")
    print(
        f"Total tid={elapsed/60:.1f} min | decode={decode_time/60:.1f} min | "
        f"process={process_time/60:.1f} min | save={save_time/60:.1f} min"
    )


if __name__ == "__main__":
    main()
