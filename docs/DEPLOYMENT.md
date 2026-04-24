# Meta-R2: Complete HuggingFace Deployment Guide (Option A)

> This guide walks you through every single step to deploy Meta-R2 to HuggingFace using the cleanest architecture:
> - **Your trained model (500MB)** → uploaded as a **HuggingFace Model Repository**
> - **Your code + environment** → deployed as a **HuggingFace Space** (Docker)
>
> The Space will auto-download the model from the Model Repo at startup. No Git LFS. No 500MB in your code repo.

---

## 🗺️ Architecture Overview

```
HuggingFace
├── YOUR-USERNAME/lifestack-agent      ← Model Repo (the 500MB weights)
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── model.safetensors  (or pytorch_model.bin)
│
└── YOUR-USERNAME/meta-r2  [SPACE]     ← Code Repo (Docker Space)
    ├── Dockerfile                       (already exists ✅)
    ├── requirements.txt                 (already exists ✅)
    ├── app_flask.py                     (entry point ✅)
    ├── core/ agent/ scripts/ ...        (all your code ✅)
    └── openenv.yaml                     (already exists ✅)
          ↓ at startup
    agent.py calls AutoModelForCausalLM.from_pretrained("YOUR-USERNAME/lifestack-agent")
    → HuggingFace downloads the model to the Space's /root/.cache/huggingface/
```

---

## ✅ Pre-Flight Checklist (Do These Before Anything Else)

Go through every item below before starting the upload steps.

### 1. Confirm Your Trained Model Files Exist

Unzip the 500MB file from Kaggle. Open the folder. You **must** see these files:

```
lifestack_model/
├── config.json              ← REQUIRED
├── tokenizer.json           ← REQUIRED
├── tokenizer_config.json    ← REQUIRED
├── special_tokens_map.json  ← REQUIRED (may be missing — check below)
└── model.safetensors        ← REQUIRED (the big file)
    OR
└── pytorch_model.bin        ← (alternative format, also fine)
```

> **If any of these are missing**, the model is an incomplete checkpoint. Re-download or re-run training with `save_model=True` at the end of `train_trl.py`.

### 2. Confirm `requirements.txt` Is Correct

Your `requirements.txt` already has:
- `openenv-core>=0.2.3` ✅ (latest version, confirmed)
- `pydantic>=2.7.0` ✅
- `transformers>=4.40.0` ✅ (needed to download model from Hub)
- `torch>=2.0.0` ✅

**No changes needed** to `requirements.txt`.

### 3. Confirm the `Dockerfile` Entry Point

Your `Dockerfile` already runs:
```dockerfile
CMD ["python", "app_flask.py"]
```
This is correct. `app_flask.py` is the web server.

**No changes needed** to the `Dockerfile`.

### 4. Make Sure `.env` is in `.gitignore`

Check your `.gitignore` — it already has:
```
.env
```
✅ Your `GROQ_API_KEY` will **never** be pushed to GitHub or HuggingFace by accident.

### 5. Make the One Required Code Change in `agent.py`

This is the only code edit required for Option A.

Open `/Users/dayalgupta/Desktop/Meta-R2/agent/agent.py` and find **lines 13–18**:

```python
# CURRENT CODE (lines 13-18):
self.api_key = os.getenv('GROQ_API_KEY')
self.local_model_path = local_model_path or os.getenv('LIFESTACK_MODEL_PATH')

# Fallback to current directory if default existence
if not self.local_model_path and os.path.exists("./lifestack_model"):
    self.local_model_path = "./lifestack_model"
```

**Change it to this** (replace `YOUR-USERNAME` with your actual HuggingFace username):

```python
# UPDATED CODE:
self.api_key = os.getenv('GROQ_API_KEY')
self.local_model_path = local_model_path or os.getenv('LIFESTACK_MODEL_PATH')

# 1. Check for local folder (Kaggle / local dev)
if not self.local_model_path and os.path.exists("./lifestack_model"):
    self.local_model_path = "./lifestack_model"

# 2. Fall back to HuggingFace Hub model repo (production / Space deployment)
if not self.local_model_path:
    self.local_model_path = "YOUR-USERNAME/lifestack-agent"
```

**Why this works:** `AutoModelForCausalLM.from_pretrained()` (which already exists on line 41) accepts either a local folder path OR a HuggingFace Hub repo ID like `"username/repo-name"`. No other code change is needed.

### 6. Verify `lifestack_model/` Is NOT in Your Code Repo

Your model (500MB) should NOT be in the `Meta-R2` GitHub repository. Confirm:
```bash
ls /Users/dayalgupta/Desktop/Meta-R2/lifestack_model/
# Should print: "No such file or directory" OR "Empty directory"
```
If it has files, remove them:
```bash
rm -rf /Users/dayalgupta/Desktop/Meta-R2/lifestack_model/*
```
The folder can stay (it's referenced in the code) but must be empty.

---

## 📦 PART 1: Upload the Model to HuggingFace Hub

### Step 1.1 — Create a HuggingFace Account

Go to **https://huggingface.co** → click **Sign Up** → create your account. Remember your username (e.g., `dayal-gupta`) — you will use it everywhere.

### Step 1.2 — Create a New Model Repository

1. Go to **https://huggingface.co/new** (or click the `+` button → "New Model")
2. Fill in:
   - **Owner:** your username
   - **Model name:** `lifestack-agent` (this becomes `YOUR-USERNAME/lifestack-agent`)
   - **License:** `MIT` (recommended for hackathons)
   - **Visibility:** `Public` (required for the Space to download it without auth)
3. Click **Create Model**

You now have an empty model repo at `https://huggingface.co/YOUR-USERNAME/lifestack-agent`.

### Step 1.3 — Install the HuggingFace CLI

On your Mac terminal:
```bash
pip install huggingface_hub
huggingface-cli login
```

When prompted, go to **https://huggingface.co/settings/tokens** → click **New token** → name it anything → **Role: Write** → copy the token → paste it into the terminal.

### Step 1.4 — Upload the Model Files

Navigate to where your unzipped model folder is (e.g., Desktop) and run:

```bash
# Replace the path with wherever your unzipped model folder is:
huggingface-cli upload YOUR-USERNAME/lifestack-agent /path/to/your/lifestack_model/ .
```

**Example (if you unzipped on Desktop):**
```bash
huggingface-cli upload dayal-gupta/lifestack-agent /Users/dayalgupta/Desktop/lifestack_model/ .
```

This uploads ALL files from the local folder to the root of the HF repo. The `.` at the end means "upload to the root of the repo."

**This will take 3–8 minutes** for a 500MB file on a normal connection. You'll see a progress bar.

### Step 1.5 — Verify the Upload

Go to `https://huggingface.co/YOUR-USERNAME/lifestack-agent` in your browser.

You should see all files listed: `config.json`, `tokenizer.json`, `model.safetensors`, etc.

Click on `config.json` and confirm it contains `"model_type"` — this confirms the model is valid and complete.

### Step 1.6 — Add a Model Card (Optional but Impressive for Judges)

Click the **"Model Card"** tab on your repo page → click the pencil icon to edit → paste this:

```markdown
---
language: en
license: mit
tags:
  - reinforcement-learning
  - life-simulation
  - grpo
  - llama
  - openenv
---

# LifeStack Agent — GRPO Fine-tuned

This model is the trained agent for [Meta-R2](https://huggingface.co/spaces/YOUR-USERNAME/meta-r2),
a reinforcement learning environment that simulates complex real-life decision-making scenarios.

Fine-tuned using GRPO (Group Relative Policy Optimization) via TRL on a custom reward function
spanning 23 life metrics across 6 domains: career, finances, relationships, physical health,
mental wellbeing, and time management.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("YOUR-USERNAME/lifestack-agent")
tokenizer = AutoTokenizer.from_pretrained("YOUR-USERNAME/lifestack-agent")
```
```

Click **Save**.

---

## 🚀 PART 2: Deploy the Project as a HuggingFace Space

### Step 2.1 — Create a New Space

1. Go to **https://huggingface.co/new-space**
2. Fill in:
   - **Owner:** your username
   - **Space name:** `meta-r2`
   - **License:** `MIT`
   - **SDK:** Select **"Docker"** ← very important, NOT Gradio or Streamlit
   - **Visibility:** `Public`
3. Click **Create Space**

You now have an empty Space at `https://huggingface.co/spaces/YOUR-USERNAME/meta-r2`.

### Step 2.2 — Connect Your GitHub Repository to the Space

This is the cleanest method — HuggingFace will auto-sync from your GitHub repo.

1. In your Space, click the **Settings** tab (gear icon)
2. Scroll down to **"Repository"** section
3. Click **"Link to a GitHub repository"**
4. Authorize HuggingFace to access your GitHub
5. Select the repo: `oki-dokii/Meta-R2`
6. Set branch: `main`
7. Click **Save**

Now every `git push` to `main` will automatically redeploy the Space. 

**Alternative (manual push):** If you don't want to link GitHub, you can push directly to the HuggingFace Space repo:

```bash
cd /Users/dayalgupta/Desktop/Meta-R2

# Add HF Space as a second remote:
git remote add space https://huggingface.co/spaces/YOUR-USERNAME/meta-r2

# Push your code:
git push space main
```

### Step 2.3 — Add the `GROQ_API_KEY` Secret to the Space

Your app needs the Groq API key at runtime. **Never hardcode it.** HuggingFace Spaces have a Secrets system for this.

1. In your Space, click the **Settings** tab
2. Scroll down to **"Variables and secrets"**
3. Click **"New secret"**
4. Fill in:
   - **Name:** `GROQ_API_KEY`
   - **Value:** your actual Groq API key (get it from https://console.groq.com/keys)
5. Click **Save**

Your `agent.py` already reads this via `os.getenv('GROQ_API_KEY')` ✅ — no code change needed.

### Step 2.4 — Add `HF_TOKEN` Secret (Required to Download the Private Model)

If your model repo is **Public** (which we set in Step 1.2), you can **skip this step**.

If your model repo is **Private**, add another secret:
- **Name:** `HF_TOKEN`
- **Value:** your HuggingFace write token (same one from Step 1.3)

Then add this line at the top of `app_flask.py` (before any model-loading code):
```python
import os
from huggingface_hub import login
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
```

### Step 2.5 — Trigger the First Build

After pushing your code (Step 2.2), the Space will automatically start building.

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR-USERNAME/meta-r2`
2. Click the **"App"** tab — you'll see a build log
3. The build will take **3–5 minutes** for the first time (Docker pulls base image, installs packages)
4. After build, it will show **"Running"** status — then the app will boot

**During the first boot**, the Space will call `AutoModelForCausalLM.from_pretrained("YOUR-USERNAME/lifestack-agent")` which will download the 500MB model. This takes about 60–90 seconds on HuggingFace infrastructure. **After the first boot, it is cached** and subsequent restarts are instant.

---

## 🔍 PART 3: Verify Everything is Working

### Step 3.1 — Check the Build Log

In your Space, click **"Logs"** tab. You should see:

```
✅ Step 1/7 : FROM python:3.11-slim
✅ Successfully built ...
✅ Successfully tagged ...
```

If you see a red error, check the troubleshooting section below.

### Step 3.2 — Check the App Boot Log

After the build, click the **"App"** tab. In the log output you should see:

```
📦 Loading local GRPO model from YOUR-USERNAME/lifestack-agent...
✅ Local model LOADED.
 * Running on http://0.0.0.0:7860
```

If you see `⚠️ Failed to load local model ... Falling back to Groq.` — the model download failed. Check that your HF model repo URL is correct in `agent.py` and the repo is public.

### Step 3.3 — Test the Live App

Go to `https://huggingface.co/spaces/YOUR-USERNAME/meta-r2` and click through the demo:
1. The web UI (served by `app_flask.py`) should load
2. Start an episode — the agent should respond with life decisions
3. Check that rewards are non-zero and steps > 5 (confirms the Task system is working)

---

## 🛠️ Troubleshooting Common Issues

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: openenv` | Wrong package in requirements.txt | Confirm `openenv-core>=0.2.3` is in `requirements.txt` (not `openenv`) |
| `OSError: Can't load model` | Wrong repo ID in `agent.py` | Make sure it's `"YOUR-ACTUAL-USERNAME/lifestack-agent"` not literally `YOUR-USERNAME` |
| `Build failed: torch install timeout` | `torch>=2.0.0` is huge (2GB+) | Add `--extra-index-url https://download.pytorch.org/whl/cpu` to Dockerfile before pip install |
| `Port 7860 not responding` | `app_flask.py` binding to wrong interface | Confirm `app.run(host='0.0.0.0', port=7860)` at the bottom of `app_flask.py` |
| `GROQ_API_KEY not found` | Secret not set | Go to Space Settings → Variables and secrets → add `GROQ_API_KEY` |
| `Space keeps restarting` | Out of memory (free tier is 16GB RAM) | torch on CPU for 500MB model may OOM — see "Reducing Memory" note below |

### Reducing Memory Usage (If Space OOMs)

Free HuggingFace Spaces have 16GB RAM. Loading a 500MB model in float32 uses ~2GB RAM, which is fine. But if you face OOM, add this to `agent.py` line 41–44:

```python
self.local_model = AutoModelForCausalLM.from_pretrained(
    self.local_model_path,
    torch_dtype=torch.float16,    # ← half precision, halves memory
    low_cpu_mem_usage=True,       # ← stream-loads, avoids peak RAM spike
    device_map="cpu"              # ← explicitly CPU on free tier
)
```

---

## 📋 Final Pre-Submission Checklist

Before submitting to the hackathon, verify every item:

- [ ] `https://huggingface.co/YOUR-USERNAME/lifestack-agent` exists and has all model files
- [ ] `https://huggingface.co/spaces/YOUR-USERNAME/meta-r2` shows **"Running"** status (green dot)
- [ ] The Space app loads in browser without errors
- [ ] The Space log shows `✅ Local model LOADED` (not "Falling back to Groq")
- [ ] An episode runs and produces steps > 5 (confirms Task system is working)
- [ ] `GROQ_API_KEY` secret is set in Space settings (as fallback)
- [ ] The model repo has a Model Card explaining what it is
- [ ] Your `README.md` in the code repo links to both: the Space URL and the Model URL
- [ ] `agent.py` has been updated with `"YOUR-USERNAME/lifestack-agent"` as the HF Hub fallback
- [ ] `lifestack_model/` folder in your local `Meta-R2/` repo is empty (model not in code repo)
- [ ] All Bugs 1, 2, 3 are fixed and committed (they are — we did this already ✅)

---

## 📎 Quick Reference — All URLs

Replace `YOUR-USERNAME` with your HuggingFace username everywhere:

| What | URL |
|---|---|
| HuggingFace profile | `https://huggingface.co/YOUR-USERNAME` |
| Model repo | `https://huggingface.co/YOUR-USERNAME/lifestack-agent` |
| Space (live demo) | `https://huggingface.co/spaces/YOUR-USERNAME/meta-r2` |
| Space settings (secrets) | `https://huggingface.co/spaces/YOUR-USERNAME/meta-r2/settings` |
| Space build logs | `https://huggingface.co/spaces/YOUR-USERNAME/meta-r2` → Logs tab |
| HuggingFace API tokens | `https://huggingface.co/settings/tokens` |
| Groq API keys | `https://console.groq.com/keys` |

---

## ⚡ The Exact Commands to Run Right Now (In Order)

```bash
# 1. Install HF CLI
pip install huggingface_hub

# 2. Login (will prompt for token)
huggingface-cli login

# 3. Upload model (change the path to your unzipped model folder)
huggingface-cli upload YOUR-USERNAME/lifestack-agent /path/to/lifestack_model/ .

# 4. Make the agent.py code change (edit manually in VS Code, then):
cd /Users/dayalgupta/Desktop/Meta-R2
git add agent/agent.py
git commit -m "feat: add HuggingFace Hub model fallback for Option A deployment"
git push origin main

# 5. Push to HuggingFace Space (if not using GitHub auto-sync):
git remote add space https://huggingface.co/spaces/YOUR-USERNAME/meta-r2
git push space main
```

That's it. The Space will build and boot automatically.
