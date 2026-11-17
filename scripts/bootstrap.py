"""Fetch exact upstream revisions without editing installed/global packages."""

import argparse, subprocess
from _common import ROOT, load


def checkout(name, destination):
    pin = load(ROOT / "manifests/upstream.json")[name]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        subprocess.run(["git", "clone", pin["url"], str(destination)], check=True)
    head = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    if head.returncode == 0 and head.stdout.strip() == pin["commit"]:
        return
    dirty = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError(f"Refusing to replace modified checkout {destination}")
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "origin", pin["commit"]], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", pin["commit"]],
        check=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training", action="store_true")
    a = p.parse_args()
    checkout("sim2real", ROOT / "third_party/sim2real")
    if a.training:
        dest = ROOT / "third_party/active-adaptation"
        checkout("active-adaptation", dest)
        checkout("mimic-lite", dest / "projects/mimic-lite")
    print("Pinned source checkouts ready.")


if __name__ == "__main__":
    main()
