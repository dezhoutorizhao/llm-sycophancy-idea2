import argparse
import torch


def load_tensor(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    for k in ["results", "R", "scores", "scores_delta"]:
        if k in obj and isinstance(obj[k], torch.Tensor):
            return obj[k].float()
    raise ValueError(f"no score tensor in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", required=True)
    ap.add_argument("--fg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ci = load_tensor(args.ci)
    fg = load_tensor(args.fg)
    if ci.shape != fg.shape:
        raise ValueError(f"shape mismatch: {ci.shape} vs {fg.shape}")
    r = 0.5 * ci + 0.5 * fg
    torch.save({"results": r}, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
