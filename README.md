# DiagramQAVisionModel

This project implements a **Diagram-Based Question Answering (DiagramQA)** system fine-tuned on the **SPIQA** dataset using the LLaVA-1.5 vision-language model. The model is trained with **LoRA (Low-Rank Adaptation)** to efficiently adapt to the diagram question-answering domain.

It takes as input a scientific diagram and a natural language question and generates a reasoned answer based on visual and textual understanding.

---

## 🧠 Objective

The goal is to enable vision-language models to interpret scientific illustrations and answer open-ended natural language questions about them — a key task in science education and automated assessment tools.

---

## 🏋️‍♂️ LoRA Fine-Tuning on SPIQA

This project fine-tunes the base **LLaVA-1.5-7B** model using **LoRA adapters** and the **Science Process Illustration Question Answering (SPIQA)** dataset.

### 📚 Dataset

- **SPIQA**: A dataset of science diagram-question-answer triplets, focused on K–12 domains like biology, earth science, and physics.
- Each sample includes:
  - A **diagram image**
  - A **natural language question**
  - A **ground-truth answer**

### ⚙️ Training Details

- **Base Model**: [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- **LoRA Framework**: [Hugging Face PEFT](https://github.com/huggingface/peft)
- **Training Strategy**:
  - Epochs: `3` *(update if different)*
  - Batch size: `8` *(update if different)*
  - Learning rate: `5e-5` *(update if different)*
  - LoRA rank: `8`
  - Optimizer: `AdamW`
- **Output**:
  - LoRA adapter weights: `adapter_model.safetensors`
  - Saved at: `/content/drive/MyDrive/model/llava-lora-reasoning/`

Only the adapter layers are updated during training, enabling cost-effective fine-tuning.

---

## 🧪 Inference Example

Once trained, the model is used for visual question answering:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(model, "/content/drive/MyDrive/model/llava-lora-reasoning/adapter_model.safetensors")
