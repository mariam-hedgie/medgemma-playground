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
*(Note: these are `.py` files but work the same)*

### **`medgemma_hf_example.py`**

A reference implementation based on Hugging Face’s official MedGemma examples.  
Demonstrates:

- Loading the MedGemma processor and model  
- Running initial inference on chest X-rays  
- Ensuring a clean **PyTorch-only** environment (TF disabled)  

---

### **`medgemma_fine_tuning_intro.py`**

Introduces the core fine-tuning workflow for MedGemma 3, including:

- Extracting the **SigLIP vision tower** and **Gemma3 language model**  
- Freezing the vision encoder to preserve pretrained radiographic features  
- Injecting **LoRA adapters** (`q_proj` / `v_proj`) into the text decoder  
- Verifying the correct trainable parameter footprint (~**0.15%** of total)  

This script establishes the architectural hooks needed for downstream multimodal alignment and text-conditioned generation experiments.

---
