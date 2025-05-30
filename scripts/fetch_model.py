import whisper, shutil, pathlib, os

dst = pathlib.Path("models")
dst.mkdir(exist_ok=True)

# Load once to make sure it's cached locally …
model = whisper.load_model("small")

# Find the cached file
cache_dir = pathlib.Path.home() / ".cache" / "whisper"
for f in cache_dir.glob("*.bin"):
    if "small" in f.name:
        shutil.copy2(f, dst / f.name)
        print("Copied", f.name, "→ models/")
