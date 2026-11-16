"""Ship the fine-tuned FT-C weights inside git without a single huge file.

GitHub rejects files over 100 MB and warns above 50 MB; the weights are about 91 MB. They are stored as three parts
of about 30 MB in demo/weights/ together with a manifest (SHA-256 of every part and of the whole file). The demo joins
them on first start and checks the hash, so a damaged or wrong file is refused instead of silently used.
Standard library only.

  python demo/weights_pack.py split demo/cache/ftc_seed20.pt     # make the parts + manifest (done once, by the authors)
  python demo/weights_pack.py join  demo/weights/ftc_seed20.json demo/cache/ftc_seed20.pt   # what the demo does itself
  python demo/weights_pack.py check                               # for users: is every part intact? prints the fix
"""
import hashlib, json, sys
from pathlib import Path

PARTS = 3


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def split(src, out_dir, parts=PARTS, note=""):
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    size = -(-len(data) // parts)
    man = {"file": src.name, "bytes": len(data), "sha256": sha256(src), "note": note, "parts": []}
    for i in range(parts):
        name = f"{src.name}.part{i + 1}"
        (out_dir / name).write_bytes(data[i * size:(i + 1) * size])
        man["parts"].append({"name": name, "bytes": len(data[i * size:(i + 1) * size]), "sha256": sha256(out_dir / name)})
    mpath = out_dir / (src.stem + ".json")
    mpath.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return mpath


def join(manifest, dst):
    """Rebuild the file next to the manifest's parts; raises if any hash does not match."""
    manifest, dst = Path(manifest), Path(dst)
    man = json.loads(manifest.read_text(encoding="utf-8"))
    buf = bytearray()
    for p in man["parts"]:
        f = manifest.parent / p["name"]
        if not f.exists():
            raise FileNotFoundError(f"missing weight part: {f}")
        if sha256(f) != p["sha256"]:
            raise ValueError(f"{f.name} is damaged (hash mismatch); download it again")
        buf += f.read_bytes()
    if hashlib.sha256(buf).hexdigest() != man["sha256"]:
        raise ValueError("the joined weights do not match the manifest hash")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(bytes(buf))
    tmp.replace(dst)
    return dst


def check(weights_dir=None, cache_dir=None) -> int:
    """Report every part as OK / MISSING / DAMAGED and print the exact commands to repair it. Returns 0 if all is well."""
    here = Path(__file__).resolve().parent
    wdir = Path(weights_dir) if weights_dir else here / "weights"
    cdir = Path(cache_dir) if cache_dir else here / "cache"
    bad = 0
    mans = sorted(wdir.glob("*.json"))
    if not mans:
        print(f"no manifest found in {wdir} (the weights folder is missing: re-download the repository)")
        return 1
    for mp in mans:
        man = json.loads(mp.read_text(encoding="utf-8"))
        print(f"== {man['file']}  ({man['bytes'] / 1e6:.1f} MB, {len(man['parts'])} parts)")
        for p in man["parts"]:
            f = wdir / p["name"]
            if not f.exists():
                print(f"  MISSING  {p['name']}"); bad += 1
            elif f.stat().st_size != p["bytes"] or sha256(f) != p["sha256"]:
                why = f"size {f.stat().st_size} instead of {p['bytes']}" if f.stat().st_size != p["bytes"] else "content does not match its hash"
                print(f"  DAMAGED  {p['name']}  ({why})"); bad += 1
            else:
                print(f"  OK       {p['name']}")
        joined = cdir / man["file"]
        if joined.exists():
            ok = joined.stat().st_size == man["bytes"] and sha256(joined) == man["sha256"]
            print(f"  {'OK      ' if ok else 'DAMAGED'}  joined file {joined}")
            if not ok:
                bad += 1
                print(f"           delete it so that it is rebuilt:   {'del' if sys.platform.startswith('win') else 'rm'} \"{joined}\"")
    if bad:
        print("\nTo repair (pick one):")
        print("  1) cloned with git:    git checkout -- demo/weights          (restores every part from your clone)")
        print("                         if that does not help:  git fetch origin && git checkout origin/main -- demo/weights")
        print("  2) downloaded a zip:   download the repository again and copy the demo/weights folder over yours")
        print("  3) skip the weights:   the demo can fine-tune its own FT-C (about 70 s on a GPU): delete the damaged files in")
        print("                         demo/weights and run  python demo/server.py  (it says so and trains from scratch)")
        print("Then run this check again:  python demo/weights_pack.py check")
        return 1
    print("\nall weight files are intact")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "split":
        print("wrote", split(sys.argv[2], Path(__file__).parent / "weights",
                             note="FT-C (all-MiniLM-L6-v2 fine-tuned on the dev split of the seed in the file name)"))
    elif len(sys.argv) >= 2 and sys.argv[1] == "check":
        sys.exit(check())
    elif len(sys.argv) >= 4 and sys.argv[1] == "join":
        print("wrote", join(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
