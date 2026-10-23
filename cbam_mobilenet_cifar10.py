#!/usr/bin/env python3
"""
MobileNet (V1, versi CIFAR-10) + CBAM  --  eksperimen posisi CBAM & dampak komputasi
Framework : TensorFlow 2.x / Keras 3 (sama dengan notebook referensi hands_on_cbam.ipynb)

Mode
  complexity : hitung parameter, MACs, latensi inferensi (TANPA training, ~1 menit)
  single     : latih 1 konfigurasi
  run_all    : latih semua konfigurasi x semua seed -> results/results.csv
  report     : agregasi results.csv -> results/report.md + results/curves.png

Contoh (Colab GPU):
  !python cbam_mobilenet_cifar10.py --mode complexity
  !python cbam_mobilenet_cifar10.py --mode run_all --epochs 30 --seeds 0 1 2
  !python cbam_mobilenet_cifar10.py --mode report

Sintaks konfigurasi:  "<posisi>[@<blok>]"
  posisi : none | before_dw | after_dw | after_pw
  blok   : all (default) | early (0-3) | mid (4-8) | late (9-12) | daftar "0,3,7"
  contoh : none after_dw after_pw after_pw@late
"""
import argparse
import csv
import json
import math
import os
import statistics
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops

# --------------------------------------------------------------------------
# 1. CBAM  (Channel Attention -> Spatial Attention, sesuai paper Woo et al. 2018)
# --------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="cbam")
class CBAM(layers.Layer):
    def __init__(self, reduction=8, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.h, self.w, self.ch = input_shape[1], input_shape[2], input_shape[3]
        hidden = max(self.ch // self.reduction, 4)
        self.hidden = hidden
        # Shared MLP (dipakai bersama oleh jalur avg-pool & max-pool)
        self.mlp1 = layers.Dense(hidden, activation="relu", name="mlp_fc1")
        self.mlp2 = layers.Dense(self.ch, name="mlp_fc2")
        # Spatial attention: Conv 7x7 pada [avg_c ; max_c]
        self.sa_conv = layers.Conv2D(1, self.kernel_size, padding="same",
                                     use_bias=False, name="sa_conv")
        super().build(input_shape)

    def call(self, x):
        # --- Channel attention ---
        avg = ops.mean(x, axis=[1, 2], keepdims=True)
        mx = ops.max(x, axis=[1, 2], keepdims=True)
        ca = ops.sigmoid(self.mlp2(self.mlp1(avg)) + self.mlp2(self.mlp1(mx)))
        x = x * ca
        # --- Spatial attention ---
        s = ops.concatenate([ops.mean(x, axis=-1, keepdims=True),
                             ops.max(x, axis=-1, keepdims=True)], axis=-1)
        sa = ops.sigmoid(self.sa_conv(s))
        return x * sa

    def macs(self):
        c, h = self.ch, self.hidden
        mlp = 2 * (c * h + h * c)                       # 2 jalur (avg & max)
        sa = self.h * self.w * self.kernel_size ** 2 * 2  # conv 7x7, 2 -> 1 channel
        return mlp + sa

    def get_config(self):
        cfg = super().get_config()
        cfg.update(reduction=self.reduction, kernel_size=self.kernel_size)
        return cfg


# --------------------------------------------------------------------------
# 2. MobileNetV1 untuk CIFAR-10 (input 32x32, stem stride 1)
# --------------------------------------------------------------------------
# (filters, stride) -- 13 blok depthwise-separable
MOBILENET_CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
                 (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
                 (1024, 2), (1024, 1)]
BLOCK_GROUPS = {"all": list(range(13)), "early": [0, 1, 2, 3],
                "mid": [4, 5, 6, 7, 8], "late": [9, 10, 11, 12]}


def parse_config(name):
    pos, _, blk = name.partition("@")
    assert pos in ("none", "before_dw", "after_dw", "after_pw"), pos
    blk = blk or "all"
    blocks = BLOCK_GROUPS[blk] if blk in BLOCK_GROUPS else [int(i) for i in blk.split(",")]
    return pos, blocks


def ds_block(x, filters, stride, idx, pos, use_cbam, reduction):
    """Blok depthwise-separable: DW3x3 -> BN -> ReLU6 -> PW1x1 -> BN -> ReLU6
    CBAM disisipkan sesuai `pos`."""
    cb = lambda t: CBAM(reduction=reduction, name=f"cbam_b{idx}")(t)
    if use_cbam and pos == "before_dw":
        x = cb(x)
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False,
                               name=f"b{idx}_dw")(x)
    x = layers.BatchNormalization(name=f"b{idx}_dw_bn")(x)
    x = layers.ReLU(6.0, name=f"b{idx}_dw_relu")(x)
    if use_cbam and pos == "after_dw":
        x = cb(x)
    x = layers.Conv2D(filters, 1, use_bias=False, name=f"b{idx}_pw")(x)
    x = layers.BatchNormalization(name=f"b{idx}_pw_bn")(x)
    x = layers.ReLU(6.0, name=f"b{idx}_pw_relu")(x)
    if use_cbam and pos == "after_pw":
        x = cb(x)
    return x


def build_mobilenet(config="none", alpha=1.0, reduction=8, num_classes=10):
    pos, blocks = parse_config(config)
    ch = lambda c: max(8, int(c * alpha))
    inp = keras.Input((32, 32, 3))
    x = layers.Conv2D(ch(32), 3, padding="same", use_bias=False, name="stem")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(6.0, name="stem_relu")(x)
    for i, (f, s) in enumerate(MOBILENET_CFG):
        x = ds_block(x, ch(f), s, i, pos, (pos != "none" and i in blocks), reduction)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, name="fc")(x)  # logits
    return keras.Model(inp, out, name=f"mobilenet_{config.replace('@', '_')}")


# --------------------------------------------------------------------------
# 3. Metrik kompleksitas: parameter, MACs, latensi
# --------------------------------------------------------------------------
def count_macs(model):
    total = 0
    for l in model.layers:
        if isinstance(l, CBAM):
            total += l.macs()
        elif isinstance(l, layers.DepthwiseConv2D):
            _, h, w, c = l.output.shape
            k = l.kernel_size[0]
            total += h * w * c * k * k
        elif isinstance(l, layers.Conv2D):
            _, h, w, co = l.output.shape
            ci = l.input.shape[-1]
            k = l.kernel_size[0]
            total += h * w * co * ci * k * k
        elif isinstance(l, layers.Dense):
            total += l.input.shape[-1] * l.units
    return total


def bench_latency(model, batch, warmup=20, iters=100):
    x = tf.random.normal((batch, 32, 32, 3))
    fn = tf.function(lambda t: model(t, training=False))
    for _ in range(warmup):
        fn(x).numpy()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x).numpy()  # .numpy() memaksa sinkronisasi (penting di GPU)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), statistics.pstdev(ts)


def complexity(model, iters=100):
    p = model.count_params()
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    m1, s1 = bench_latency(model, 1, iters=iters)
    mb, sb = bench_latency(model, 256, iters=max(10, iters // 5))
    return dict(params=p, trainable=trainable, macs=count_macs(model),
                lat_b1_ms=m1, lat_b1_std=s1,
                lat_b256_ms_per_img=mb / 256, lat_b256_std=sb / 256)


# --------------------------------------------------------------------------
# 4. Data CIFAR-10
# --------------------------------------------------------------------------
def load_data(batch, seed, smoke=False):
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    ytr, yte = ytr.squeeze(), yte.squeeze()
    if smoke:
        xtr, ytr, xte, yte = xtr[:1500], ytr[:1500], xte[:500], yte[:500]
    # split train/val TETAP (seed 1234) agar semua konfigurasi & seed memakai data sama
    perm = np.random.RandomState(1234).permutation(len(xtr))
    n_val = 500 if smoke else 5000
    vi, ti = perm[:n_val], perm[n_val:]
    mean = np.array([0.4914, 0.4822, 0.4465], "float32")
    std = np.array([0.2470, 0.2435, 0.2616], "float32")
    norm = lambda a: ((a.astype("float32") / 255.0) - mean) / std
    aug = keras.Sequential([layers.RandomFlip("horizontal"),
                            layers.RandomTranslation(0.125, 0.125, fill_mode="reflect")])
    AT = tf.data.AUTOTUNE
    train = (tf.data.Dataset.from_tensor_slices((norm(xtr[ti]), ytr[ti]))
             .shuffle(len(ti), seed=seed).batch(batch)
             .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AT).prefetch(AT))
    mk = lambda x, y: tf.data.Dataset.from_tensor_slices((norm(x), y)).batch(256).prefetch(AT)
    return train, mk(xtr[vi], ytr[vi]), mk(xte, yte), len(ti)


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.t = []
    def on_epoch_begin(self, e, logs=None): self._s = time.perf_counter()
    def on_epoch_end(self, e, logs=None): self.t.append(time.perf_counter() - self._s)


# --------------------------------------------------------------------------
# 5. Train satu konfigurasi + satu seed
# --------------------------------------------------------------------------
def run_one(config, seed, args):
    keras.utils.set_random_seed(seed)
    train, val, test, n_train = load_data(args.batch, seed, args.smoke)
    model = build_mobilenet(config, args.alpha, args.reduction)
    steps = math.ceil(n_train / args.batch) * args.epochs
    lr = keras.optimizers.schedules.CosineDecay(args.lr, steps)
    model.compile(keras.optimizers.AdamW(lr, weight_decay=5e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    tag = f"{config.replace('@', '_')}_s{seed}"
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"{tag}.weights.h5")
    timer = EpochTimer()
    hist = model.fit(train, validation_data=val, epochs=args.epochs, verbose=2,
                     callbacks=[timer, keras.callbacks.ModelCheckpoint(
                         ckpt, monitor="val_accuracy", save_best_only=True,
                         save_weights_only=True)])
    model.load_weights(ckpt)  # bobot epoch terbaik (berdasar VAL, bukan test)
    test_loss, test_acc = model.evaluate(test, verbose=0)
    row = dict(config=config, seed=seed, epochs=args.epochs, alpha=args.alpha,
               best_val_acc=max(hist.history["val_accuracy"]), test_acc=test_acc,
               test_loss=test_loss, sec_per_epoch=float(np.mean(timer.t)))
    row.update(complexity(model, iters=50))
    json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()},
              open(os.path.join(args.out, f"{tag}_history.json"), "w"))
    csv_path = os.path.join(args.out, "results.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); 
        if new: w.writeheader()
        w.writerow(row)
    print(f"[DONE] {tag}  val={row['best_val_acc']:.4f}  test={test_acc:.4f}  "
          f"params={row['params']:,}")
    os.remove(ckpt)
    return row


# --------------------------------------------------------------------------
# 6. Report
# --------------------------------------------------------------------------
def make_report(out):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(os.path.join(out, "results.csv"))
    g = df.groupby("config", sort=False)
    t = g.agg(n=("seed", "count"), test_mean=("test_acc", "mean"), test_std=("test_acc", "std"),
              val_mean=("best_val_acc", "mean"), params=("params", "first"), macs=("macs", "first"),
              lat1=("lat_b1_ms", "mean"), lat256=("lat_b256_ms_per_img", "mean"),
              sec_ep=("sec_per_epoch", "mean")).fillna(0)
    base = t.loc["none"] if "none" in t.index else t.iloc[0]
    lines = ["| Konfigurasi | n | Test Acc (%) | Δ vs base (pp) | Params | Δ Params | MMACs | "
             "Latensi b=1 (ms) | Latensi b=256 (ms/img) | s/epoch |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in t.iterrows():
        lines.append(
            f"| {c} | {int(r.n)} | {100*r.test_mean:.2f} ± {100*r.test_std:.2f} | "
            f"{100*(r.test_mean-base.test_mean):+.2f} | {int(r.params):,} | "
            f"{100*(r.params/base.params-1):+.2f}% | {r.macs/1e6:.2f} | "
            f"{r.lat1:.2f} ({100*(r.lat1/base.lat1-1):+.0f}%) | "
            f"{r.lat256:.4f} ({100*(r.lat256/base.lat256-1):+.0f}%) | {r.sec_ep:.1f} |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plt.figure(figsize=(7, 4.5))
    for c in t.index:
        hs = [json.load(open(os.path.join(out, f"{c.replace('@','_')}_s{s}_history.json")))["val_accuracy"]
              for s in df[df.config == c].seed]
        plt.plot(np.mean(hs, 0), label=c)
    plt.xlabel("Epoch"); plt.ylabel("Val accuracy"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "curves.png"), dpi=150)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="complexity",
                    choices=["complexity", "single", "run_all", "report"])
    ap.add_argument("--configs", nargs="+", default=["none", "before_dw", "after_dw", "after_pw"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--alpha", type=float, default=1.0, help="width multiplier MobileNet")
    ap.add_argument("--reduction", type=int, default=8, help="reduction ratio CBAM")
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true", help="subset kecil, cek pipeline saja")
    args = ap.parse_args()

    if args.mode == "complexity":
        os.makedirs(args.out, exist_ok=True)
        rows = []
        for c in args.configs:
            m = build_mobilenet(c, args.alpha, args.reduction)
            r = dict(config=c, **complexity(m))
            rows.append(r)
            print(f"{c:16s} params={r['params']:>10,}  MMACs={r['macs']/1e6:8.2f}  "
                  f"b1={r['lat_b1_ms']:7.2f}ms  b256={r['lat_b256_ms_per_img']:.4f}ms/img")
        b = rows[0]
        print("\nOverhead vs", b["config"])
        for r in rows[1:]:
            print(f"{r['config']:16s} params {100*(r['params']/b['params']-1):+6.2f}%  "
                  f"MACs {100*(r['macs']/b['macs']-1):+6.2f}%  "
                  f"lat(b=1) {100*(r['lat_b1_ms']/b['lat_b1_ms']-1):+6.1f}%  "
                  f"lat(b=256) {100*(r['lat_b256_ms_per_img']/b['lat_b256_ms_per_img']-1):+6.1f}%")
        json.dump(rows, open(os.path.join(args.out, "complexity.json"), "w"), indent=1)
    elif args.mode == "single":
        run_one(args.configs[0], args.seeds[0], args)
    elif args.mode == "run_all":
        for s in args.seeds:
            for c in args.configs:
                run_one(c, s, args)
        make_report(args.out)
    else:
        make_report(args.out)
