# 🫁 MedGemma Chest X-Ray Playground

A lightweight experimental environment for exploring **MedGemma 3** as a multimodal backbone for chest X-ray understanding, alignment, and generation workflows.

This project focuses on practical, hands-on experimentation with:

- **Hugging Face integration**
- **Vision–language architecture inspection**
- **Parameter-efficient fine-tuning (LoRA)**
- **Prompt / image alignment fundamentals**
- **Dataset preparation for downstream generation tasks**

---

## 📘 Notebooks  
> **Note:** GitHub does not reliably render `.ipynb` files that include widget metadata (common in Colab & HF examples).  
> All notebooks run correctly when downloaded or opened in **Google Colab**.  
> If GitHub preview fails, use **“Download raw file”** and open locally or in Colab.

---

### `notebooks/medgemma_hf_example.ipynb`

A reference implementation based on Hugging Face’s official MedGemma examples.

Demonstrates:

- Loading the MedGemma processor + model  
- Running initial inference on sample CXRs  
- Ensuring a clean **PyTorch-only** environment (TensorFlow disabled)

---

### `notebooks/medgemma_fine_tuning_intro.ipynb`

Introduces the foundational fine-tuning workflow for MedGemma 3:

- Extracting the **SigLIP vision tower** and **Gemma3 language model**  
- Freezing the vision encoder  
- Injecting **LoRA adapters** (`q_proj` / `v_proj`)  
- Verifying trainable parameter count (~**0.15%** of total)  

Builds the architectural hooks needed for controlled multimodal alignment.

---

### `notebooks/medgemma_data_pipeline.ipynb`

Implements the full data ingestion pipeline required for fine-tuning:

1. Download or load sample chest X-ray images  
2. Build a **JSONL multimodal dataset** with `{"image", "prompt", "cxr_report"}`  
3. Define a custom PyTorch **Dataset**  
4. Implement a custom **collate_fn** using the MedGemma image processor & tokenizer  
5. Produce aligned tensors:
   - `pixel_values` → `(B, 3, H, W)`  
   - `input_ids`, `attention_mask` → `(B, T)`  

This notebook forms the backbone for downstream training loops.

---

## 📁 Scripts (`scripts/`)

Linearized `.py` exports of the interactive notebooks.  
These provide clean reference code but **are not intended to run standalone** (they lack notebook state, mounting logic, etc.).

- `scripts/medgemma_hf_example.py`  
- `scripts/medgemma_fine_tuning_intro.py`  
- `scripts/medgemma_data_pipeline.py`

---

## 📂 Data & JSONL

Running `notebooks/medgemma_data_pipeline.ipynb` produces:

```python
data/images/
data/cxr_generation_dataset.jsonl
```

These files are included in the repo so users don’t need to re-run the notebook unless they want to modify the dataset.

---

## 🚀 Quick Start (using pre-generated data)

If you don’t want to run the data-pipeline notebook:

1. The repo already includes sample CXRs in `data/images/`  
2. A ready-to-use dataset file is available at: data/cxr_generation_dataset.jsonl
3. Load it directly:

```python
from medgemma_data_pipeline import CXRGenerationDataset
dataset = CXRGenerationDataset("data/cxr_generation_dataset.jsonl")
sample = dataset[0]
print(sample.keys())
```

