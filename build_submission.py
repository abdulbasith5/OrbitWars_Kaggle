import tarfile, os, shutil

# Rename checkpoint to what main.py expects
src = "latest_ckpt (1).pt"
dst = "ckpt_final.pt"
if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"Checkpoint renamed: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
else:
    print(f"WARNING: {src} not found, skipping rename")

# Build submission archive
with tarfile.open("submission.tar.gz", "w:gz") as tar:
    # main.py at root — required by Kaggle
    tar.add("main.py", arcname="main.py")
    print("Added: main.py")

    # checkpoint at root (main.py searches here too)
    if os.path.exists("ckpt_final.pt"):
        tar.add("ckpt_final.pt", arcname="ckpt_final.pt")
        print("Added: ckpt_final.pt")

    # RL source modules needed at runtime
    for f in ["policy.py", "features.py", "config.py"]:
        src_path = os.path.join("rl", "src", f)
        if os.path.exists(src_path):
            tar.add(src_path, arcname=f"rl/src/{f}")
            print(f"Added: rl/src/{f}")

print()
print("=== submission.tar.gz contents ===")
with tarfile.open("submission.tar.gz", "r:gz") as tar:
    for m in tar.getmembers():
        print(f"  {m.name}  ({m.size/1000:.1f} KB)")

size_mb = os.path.getsize("submission.tar.gz") / 1e6
print(f"Total size: {size_mb:.2f} MB")
print()
print("Done! Upload submission.tar.gz to Kaggle competition.")
