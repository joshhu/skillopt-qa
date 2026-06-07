"""下載資料集並建立 SkillOpt 的 train/val/test 切分。

    # 英文多跳(HotpotQA)
    uv run skillopt-download --out data/hotpotqa --n-train 64 --n-val 64 --n-test 200
    # 繁體中文閱讀理解(DRCD)
    uv run skillopt-download --benchmark drcd --out data/drcd --n-train 64 --n-val 64 --n-test 200
"""

from __future__ import annotations

import argparse

from skillopt.data import build_drcd_split, build_split

_DEFAULT_OUT = {"hotpotqa": "data/hotpotqa", "drcd": "data/drcd"}


def main() -> None:
    p = argparse.ArgumentParser(description="Build a train/val/test split for SkillOpt")
    p.add_argument("--benchmark", default="hotpotqa",
                   choices=["hotpotqa", "drcd"],
                   help="hotpotqa(英文多跳)或 drcd(繁體中文閱讀理解)")
    p.add_argument("--out", default=None,
                   help="輸出切分的目錄(預設依 benchmark 而定)")
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hf-config", default="distractor",
                   choices=["distractor", "fullwiki"],
                   help="僅 HotpotQA 適用")
    args = p.parse_args()

    out = args.out or _DEFAULT_OUT[args.benchmark]
    if args.benchmark == "drcd":
        out = build_drcd_split(
            out,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
        )
    else:
        out = build_split(
            out,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
            hf_config=args.hf_config,
        )
    print(f"Wrote train/val/test items.json under {out}/")


if __name__ == "__main__":
    main()
