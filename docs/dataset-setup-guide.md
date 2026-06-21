# KaraAgro AI Maize Dataset — Setup & Download Guide

This guide walks through setting up Lightning AI and downloading the **KaraAgro AI Maize** dataset (Supervisely format, ~31.5GB) directly into your workspace — no local upload required.

Part of the **Angalia** project (edge AI Fall Armyworm detection for Kenyan smallholder maize farms).

---

## Part 1: Lightning AI Account Setup

1. Go to [lightning.ai](https://lightning.ai) and sign up for a free account.
2. Once logged in, create a new **Studio** — this is your cloud workspace (think of it as a VM with a terminal + notebook environment built in).
3. Inside the Studio, you'll see:
   - A **terminal** icon for running shell commands.
   - A **notebook** icon (usually on the right sidebar) for running Jupyter-style cells.
4. Your default working directory is typically `/teamspace/studios/this_studio` — no need to navigate elsewhere before starting.
5. Quick sanity check — open a terminal and run:
   ```bash
   python --version
   pip --version
   ```
   If both return version numbers without error, you're ready to go.

---

## Part 2: Downloading & Unpacking the Dataset

We use the `dataset-tools` Python package, which downloads **and** automatically unpacks the dataset in one step — server-to-server, so your local upload speed is never a bottleneck.

### Step 1 — Install the package
In the terminal:
```bash
pip install --upgrade dataset-tools
```

> **Note:** You may see a warning like:
> ```
> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...
> lightning-sdk requires urllib3<=2.5.0, but you have urllib3 2.7.0...
> scikit-learn requires numpy<2.0, but you have numpy 2.3.3...
> ```
> **This is safe to ignore.** It's a warning, not a failure. Confirm the package still works:
> ```bash
> python -c "import dataset_tools as dtools; print('ok')"
> ```
> If it prints `ok`, you're fine.

### Step 2 — Create the download script
If `nano` isn't available (`zsh: command not found: nano`), skip the editor and write the file directly using a heredoc:
```bash
cat > download.py << 'EOF'
import dataset_tools as dtools

dtools.download(dataset='KaraAgro AI Maize', dst_dir='~/dataset-ninja/')
EOF
```
Verify it saved correctly:
```bash
cat download.py
```

### Step 3 — Run it in the background
```bash
nohup python download.py > download.log 2>&1 &
```
The `&` runs it in the background so your terminal stays free. Watch progress with:
```bash
tail -f download.log
```
Press `Ctrl+C` to stop watching (download keeps running either way).

Expected output looks like:
```
Downloading 'KaraAgro AI Maize': 100%|██████████| 31.5G/31.5G [11:58<00:00, 47.1MB/s]
Unpacking 'karaagro-ai-maize.tar': 100%|██████████| 49664/49664 [03:51<00:00, 214.14file/s]
```

### Step 4 — Verify the download (in a notebook cell)

Run these one cell at a time — easier to debug than one large script.

**Cell 1 — set the path**
```python
import os
import glob

base_dir = os.path.expanduser('~/dataset-ninja')
print(base_dir)
```

**Cell 2 — confirm the folder exists**
```python
print(os.path.exists(base_dir))
print(os.listdir(base_dir))
```

**Cell 3 — count images**
```python
image_paths = glob.glob(f'{base_dir}/**/*.jpg', recursive=True) + \
              glob.glob(f'{base_dir}/**/*.png', recursive=True)
print(f"Total images found: {len(image_paths)}")
```
Expected: **16,552** images.

**Cell 4 — check folder structure (top 2 levels only)**
```python
for root, dirs, files in os.walk(base_dir):
    depth = root.replace(base_dir, '').count(os.sep)
    if depth < 2:
        print(root)
```

**Cell 5 — total size on disk**
```python
total_size = sum(os.path.getsize(os.path.join(dirpath, f))
                  for dirpath, _, filenames in os.walk(base_dir)
                  for f in filenames)
print(f"Total size: {total_size / (1024**3):.2f} GB")
```

### Step 5 — Clean up the leftover archive

> **Gotcha:** The unpacking step does **not** delete the original `.tar` file. If your total disk size comes out to roughly **double** the downloaded size (~62GB instead of ~31GB), the archive is still sitting on disk.

Find and remove it:
```python
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith('.tar'):
            full_path = os.path.join(root, f)
            print(full_path, os.path.getsize(full_path) / (1024**3), "GB")
```
Once confirmed, delete it:
```python
import os
os.remove(full_path)  # use the actual path printed above
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `nano: command not found` | Use the `cat > file << 'EOF' ... EOF` heredoc method instead of an editor |
| pip dependency conflict warning | Safe to ignore if `import dataset_tools` still works. If it breaks something critical, isolate in a virtual environment (`python -m venv dataset_env`) instead of force-downgrading packages |
| Disk size looks doubled after unpacking | The original `.tar` wasn't auto-deleted — find and remove it manually (see Step 5) |
| Download seems slow or stalls | Check connection with `ping lightning.ai` or just rerun — `dataset-tools` downloads are generally resumable |

---

## Result

After following these steps you should have:
- **16,552 images** with bounding box annotations in Supervisely format
- Organized in `~/dataset-ninja/` with per-image JSON annotation files
- ~31GB on disk after cleanup (no duplicate archive)

Ready for conversion into YOLOv8 training format for the Angalia FAW detection pipeline.
