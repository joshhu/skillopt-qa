"""下載 HotpotQA 並建立 SkillOpt 的 train/val/test 切分。

    uv run skillopt-download --out data/hotpotqa --n-train 64 --n-val 64 --n-test 200
"""

from __future__ import annotations

import argparse

from skillopt.data import build_split


def main() -> None:
    p = argparse.ArgumentParser(description="Build a HotpotQA split for SkillOpt")
    p.add_argument("--out", default="data/hotpotqa", help="輸出切分的目錄")
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hf-config", default="distractor",
                   choices=["distractor", "fullwiki"])
    args = p.parse_args()

    out = build_split(
        args.out,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        hf_config=args.hf_config,
    )
    print(f"Wrote train/val/test items.json under {out}/")


if __name__ == "__main__":
    main()
