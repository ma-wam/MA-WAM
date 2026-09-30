"""Filter a REAL dataset dir to its top-fraction by per-agent episode return.
Reads  <src>/{obs,actions,rewards,path_lengths[,legals]}.npy
Writes <dst>/{...} keeping only the highest-return episodes.
Used by R3 (train a world model on high-return real data so synthesis stays high-quality)."""
import argparse, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--keep-frac", type=float, default=0.5, help="top fraction by return to keep")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    pls = np.load(os.path.join(args.src, "path_lengths.npy"))
    rew = np.load(os.path.join(args.src, "rewards.npy"))
    n = len(pls)
    st = np.zeros(n, dtype=np.int64); st[1:] = np.cumsum(pls[:-1])
    rets = np.empty(n, dtype=np.float64)
    for i, (s, l) in enumerate(zip(st, pls)):
        er = rew[s:s + int(l)]
        rets[i] = er.sum() / er.shape[1] if er.ndim >= 2 else er.sum()
    n_keep = max(1, int(round(n * args.keep_frac)))
    keep = np.sort(np.argsort(rets)[::-1][:n_keep])   # top by return, keep original order
    # build flat row index
    rows = np.concatenate([np.arange(st[i], st[i] + int(pls[i])) for i in keep])
    for fn in ["obs.npy", "actions.npy", "rewards.npy", "legals.npy"]:
        p = os.path.join(args.src, fn)
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        np.save(os.path.join(args.dst, fn), arr[rows])
    np.save(os.path.join(args.dst, "path_lengths.npy"), pls[keep])
    print(f"[real-filter] {args.src} -> {args.dst}: kept {n_keep}/{n} "
          f"(return p50 all={np.median(rets):.3f} kept={np.median(rets[keep]):.3f} "
          f"min_kept={rets[keep].min():.3f})")

if __name__ == "__main__":
    main()
