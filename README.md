🫁 MedGemma Chest X-Ray Playground

A lightweight experimental environment for exploring MedGemma 3 as a multimodal backbone for chest X-ray understanding, alignment, and generation workflows.

This project focuses on practical, hands-on experimentation with:
	•	Hugging Face integration
	•	Vision–language architecture inspection
	•	Parameter-efficient fine-tuning (LoRA)
	•	Prompt / image alignment fundamentals
	•	Dataset preparation for downstream generation tasks

  📘 Notebooks
  (note: they are .py files here, but will work fine after downloading)

medgemma_hf_example.py

A reference implementation based on Hugging Face’s official MedGemma examples.
Includes loading the processor and model, running initial inference, and verifying the PyTorch-only setup.

medgemma_fine_tuning_intro.py

Introduces the core fine-tuning workflow:
	•	Extracting the SigLIP vision tower and Gemma3 language model
	•	Freezing the vision encoder
	•	Injecting LoRA adapters (q_proj/v_proj) into the text decoder
	•	Confirming trainable parameter count (~0.15% of total)
This notebook sets up the exact architectural hooks needed for later multimodal alignment or generation conditioning.
