"""
prepare_dataset.py
==================
Step 1: Crop faces from:
  - Real class : Kaggle anti-spoofing-live dataset  (selfie photos)
  - Spoof class: LCC_FASD dataset  (print / replay attack images)

Output layout expected by the trainers:
  data/
    train/
      real/   *.jpg
      spoof/  *.jpg
    val/
      real/   *.jpg
      spoof/  *.jpg

Usage:
  pip install opencv-python-headless retina-face  (or use your existing detector.py)
  python prepare_dataset.py
"""

import os
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ─── CONFIGURE THESE PATHS ────────────────────────────────────────────────────
KAGGLE_REAL_DIR  = r"data/raw/anti-spoofing-live"   # unzipped Kaggle dataset root
LCC_FASD_DIR     = r"data/raw/LCC_FASD"             # LCC_FASD root (has train/val/real/spoof)
CASIA_DIR        = r"data/raw/casiafasd"            # optional CASIA-FASD root
MSU_DIR          = r"data/raw/msu-mfsd"             # optional MSU-MFSD processed frames root
OUT_DIR          = r"data"                           # output root
VAL_SPLIT        = 0.15                              # 15 % for validation
MAX_REAL         = 10_000                            # cap to avoid huge imbalance
MAX_SPOOF        = 10_000
IMG_EXTS         = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXTS         = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
VIDEO_FRAME_STEP = 10                                # use every Nth frame from each video
# ──────────────────────────────────────────────────────────────────────────────

random.seed(42)

# ── face detector (uses your existing detector.py if it's on the path) ────────
def get_detector():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from IADG import aFaceDetect
    
    print("[detector] using YOLOv8-face from IADG.py")
    model = aFaceDetect()

    def detect(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bboxes, landmarks = model(img_rgb)
        if len(bboxes) == 0:
            return None
        b = bboxes[0]
        x1 = int(b[0]);  y1 = int(b[1])
        x2 = int(b[0] + b[2]);  y2 = int(b[1] + b[3])
        return x1, y1, x2, y2

    return detect


def collect_media(root, max_n=None):
    """Walk a directory and return shuffled image/video entries."""
    entries = []
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in IMG_EXTS:
            entries.append(("image", str(p)))
        elif suffix in VID_EXTS:
            entries.append(("video", str(p)))
    random.shuffle(entries)
    return entries[:max_n] if max_n else entries


def media_stats(entries):
    img_n = sum(1 for k, _ in entries if k == "image")
    vid_n = sum(1 for k, _ in entries if k == "video")
    return img_n, vid_n


def collect_lcc_class_media(root, class_name, max_n=None):
    """Collect class media from LCC_FASD across different directory layouts."""
    class_dirs = [
        str(p) for p in Path(root).rglob("*")
        if p.is_dir() and p.name.lower() == class_name.lower()
    ]
    class_paths = []
    for d in class_dirs:
        class_paths += collect_media(d)
    random.shuffle(class_paths)
    return class_paths[:max_n] if max_n else class_paths


def collect_casia_labeled_media(root, max_n=None):
    """
    Collect CASIA media and split by filename labels.
    Expected labels in filename: _real / _live and _fake / _spoof / _attack.
    """
    real_tokens = ("_real", "_live")
    spoof_tokens = ("_fake", "_spoof", "_attack")
    real_entries = []
    spoof_entries = []

    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in IMG_EXTS and suffix not in VID_EXTS:
            continue
        kind = "image" if suffix in IMG_EXTS else "video"
        name = p.name.lower()
        entry = (kind, str(p))
        if any(t in name for t in real_tokens):
            real_entries.append(entry)
        elif any(t in name for t in spoof_tokens):
            spoof_entries.append(entry)

    random.shuffle(real_entries)
    random.shuffle(spoof_entries)
    if max_n:
        real_entries = real_entries[:max_n]
        spoof_entries = spoof_entries[:max_n]
    return real_entries, spoof_entries


def collect_labeled_media_by_dir(root, real_dir_names, spoof_dir_names, max_n=None):
    """
    Collect media by parent directory names.
    Useful for datasets like MSU-MFSD where labels are folder-based.
    """
    real_set = {n.lower() for n in real_dir_names}
    spoof_set = {n.lower() for n in spoof_dir_names}
    real_entries = []
    spoof_entries = []

    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in IMG_EXTS and suffix not in VID_EXTS:
            continue
        kind = "image" if suffix in IMG_EXTS else "video"
        parent_chain = {part.lower() for part in p.parent.parts}
        entry = (kind, str(p))
        if real_set.intersection(parent_chain):
            real_entries.append(entry)
        elif spoof_set.intersection(parent_chain):
            spoof_entries.append(entry)

    random.shuffle(real_entries)
    random.shuffle(spoof_entries)
    if max_n:
        real_entries = real_entries[:max_n]
        spoof_entries = spoof_entries[:max_n]
    return real_entries, spoof_entries


def crop_and_save(entries, dst_dir, detect_fn, label):
    os.makedirs(dst_dir, exist_ok=True)
    saved = 0
    video_files = 0
    video_opened = 0
    video_frames_read = 0
    video_frames_sampled = 0
    video_frames_saved = 0

    def crop_face(img):
        box = detect_fn(img)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    for kind, src in tqdm(entries, desc=f"  {label} -> {dst_dir}"):
        if kind == "image":
            img = cv2.imread(src)
            if img is None:
                continue
            crop = crop_face(img)
            if crop is None:
                continue
            out_path = os.path.join(dst_dir, f"{saved:06d}.jpg")
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            continue

        cap = cv2.VideoCapture(src)
        video_files += 1
        if not cap.isOpened():
            continue
        video_opened += 1
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            video_frames_read += 1
            if frame_idx % VIDEO_FRAME_STEP == 0:
                video_frames_sampled += 1
                crop = crop_face(frame)
                if crop is not None:
                    out_path = os.path.join(dst_dir, f"{saved:06d}.jpg")
                    cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
                    video_frames_saved += 1
            frame_idx += 1
        cap.release()

    print(f"  saved {saved} images to {dst_dir}")
    print(
        "  video stats:"
        f" files={video_files}, opened={video_opened},"
        f" frames_read={video_frames_read}, sampled={video_frames_sampled},"
        f" saved={video_frames_saved}"
    )
    return saved

def split_into_train_val(src_dir, train_dir, val_dir, val_ratio=0.15):
    """Move files from src_dir into train_dir / val_dir."""
    files = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
    random.shuffle(files)
    n_val  = int(len(files) * val_ratio)
    val_f  = files[:n_val]
    train_f= files[n_val:]
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)
    for d in (train_dir, val_dir):
        for f in os.listdir(d):
            if f.endswith(".jpg"):
                os.remove(os.path.join(d, f))
    for f in train_f:
        shutil.move(os.path.join(src_dir, f), os.path.join(train_dir, f))
    for f in val_f:
        shutil.move(os.path.join(src_dir, f), os.path.join(val_dir, f))
    print(f"  split: {len(train_f)} train / {len(val_f)} val")


def rebalance_temp_dirs(real_dir, spoof_dir):
    """Downsample larger class so both classes have the same count."""
    real_files = [f for f in os.listdir(real_dir) if f.endswith(".jpg")]
    spoof_files = [f for f in os.listdir(spoof_dir) if f.endswith(".jpg")]
    n_real = len(real_files)
    n_spoof = len(spoof_files)
    target = min(n_real, n_spoof)

    if target == 0:
        print("[warn] Cannot rebalance: one class has 0 images after cropping.")
        return n_real, n_spoof, target

    random.shuffle(real_files)
    random.shuffle(spoof_files)

    for f in real_files[target:]:
        os.remove(os.path.join(real_dir, f))
    for f in spoof_files[target:]:
        os.remove(os.path.join(spoof_dir, f))

    print(f"[rebalance] real={n_real}, spoof={n_spoof}, using {target} per class")
    return n_real, n_spoof, target


def main():
    detect = get_detector()
    tmp_real  = os.path.join(OUT_DIR, "_tmp_real")
    tmp_spoof = os.path.join(OUT_DIR, "_tmp_spoof")

    # ── REAL: Kaggle anti-spoofing-live ───────────────────────────────────────
    print("\n[1/4] Collecting REAL media from Kaggle dataset …")
    real_paths = collect_media(KAGGLE_REAL_DIR)
    lcc_real = collect_lcc_class_media(LCC_FASD_DIR, "real")
    casia_real = []
    casia_spoof = []
    msu_real = []
    msu_spoof = []
    if os.path.isdir(CASIA_DIR):
        casia_real, casia_spoof = collect_casia_labeled_media(CASIA_DIR)
        print(f"  CASIA labeled media: real={len(casia_real)} spoof={len(casia_spoof)}")
    if os.path.isdir(MSU_DIR):
        msu_real, msu_spoof = collect_labeled_media_by_dir(
            MSU_DIR,
            real_dir_names=("real", "live", "genuine", "bonafide"),
            spoof_dir_names=("attack", "fake", "spoof", "print", "replay"),
        )
        print(f"  MSU labeled media: real={len(msu_real)} spoof={len(msu_spoof)}")
    print(f"  LCC labeled media: real={len(lcc_real)}")
    real_pool = real_paths + lcc_real + casia_real + msu_real
    random.shuffle(real_pool)
    real_pool = real_pool[:MAX_REAL]
    print(f"  found {len(real_pool)} candidate real media")
    real_img_n, real_vid_n = media_stats(real_pool)
    print(f"  real mix: images={real_img_n} videos={real_vid_n}")
    crop_and_save(real_pool, tmp_real, detect, "real")

    # ── SPOOF: LCC_FASD ───────────────────────────────────────────────────────
    # LCC_FASD already has train/val splits but we re-pool and re-split for
    # a clean ratio with our real data.
    print("\n[2/4] Collecting SPOOF media from LCC_FASD …")
    lcc_spoof = collect_lcc_class_media(LCC_FASD_DIR, "spoof")
    print(f"  LCC labeled media: spoof={len(lcc_spoof)}")
    spoof_pool = lcc_spoof + casia_spoof + msu_spoof
    random.shuffle(spoof_pool)
    spoof_pool = spoof_pool[:MAX_SPOOF]
    print(f"  found {len(spoof_pool)} candidate spoof media")
    spoof_img_n, spoof_vid_n = media_stats(spoof_pool)
    print(f"  spoof mix: images={spoof_img_n} videos={spoof_vid_n}")
    crop_and_save(spoof_pool, tmp_spoof, detect, "spoof")
    rebalance_temp_dirs(tmp_real, tmp_spoof)

    # ── SPLIT ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Splitting into train / val …")
    split_into_train_val(
        tmp_real,
        os.path.join(OUT_DIR, "train", "real"),
        os.path.join(OUT_DIR, "val",   "real"),
        VAL_SPLIT,
    )
    split_into_train_val(
        tmp_spoof,
        os.path.join(OUT_DIR, "train", "spoof"),
        os.path.join(OUT_DIR, "val",   "spoof"),
        VAL_SPLIT,
    )

    # cleanup temp dirs
    shutil.rmtree(tmp_real,  ignore_errors=True)
    shutil.rmtree(tmp_spoof, ignore_errors=True)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n[4/4] Dataset ready:")
    for split in ("train", "val"):
        for cls in ("real", "spoof"):
            d = os.path.join(OUT_DIR, split, cls)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  {split}/{cls}: {n} images")


if __name__ == "__main__":
    main()
