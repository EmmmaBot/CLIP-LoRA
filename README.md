This repository is a fork of the original implementation of the paper [*Low-Rank Few-Shot Adaptation of Vision-Language Models*](https://arxiv.org/abs/2405.18541), as presented at CVPRW 2024.

The paper introduces efficient fine-tuning approaches for vision-language models (e.g., CLIP) using Low-Rank Adaptation (LoRA). For more details, refer to the paper linked above.

---

### Note on Updates

This README and codebase have been modified from the original repository to reflect custom updates and enhancements over time. Any new features or improvements added here are based on the original implementation. 

Feel free to explore this forked repository and track changes in the commit history.

## Features

- **Available Models:** Supports variants of the CLIP model such as `RN50`, `RN101`, `ViT-B/32`, `ViT-B/16`, and `ViT-L/14`.
- **Efficient Fine-Tuning:** Use LoRA to adapt large-scale models without modifying their core weights.
- **Compatibility:** Supports CPU and GPU (CUDA) devices.
- **Comprehensive Preprocessing Pipelines:**
  - Text tokenization with fixed context length.
  - Image resizing, cropping, and normalization.

## Installation

To run this repository, you will need Python installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/EmmmaBot/CLIP-LoRA.git
cd CLIP-LoRA
pip install -r requirements.txt
```


## Paper

This implementation is based on the paper:
**"Low-Rank Few-Shot Adaptation of Vision-Language Models" (CLIP-LoRA)**  
*Presented at CVPRW 2024.*  

If you use this repository in your work, consider citing the paper.

## Acknowledgements

The implementation is inspired by OpenAI's [CLIP](https://github.com/openai/CLIP) and related work in efficient model fine-tuning techniques. Special thanks to the authors of the CLIP-LoRA paper for motivating this project.


## Run the Script

The repository includes a main script `main.py` that allows users to perform various tasks by passing parameters. Make sure you have installed the required dependencies and have the necessary dataset/models available.

### Example Usage

1. **Fine-Tuning a CLIP Model with LoRA:**
   Specify the path to your dataset, the model to fine-tune, and other parameters:

   ```bash
   python main.py --task finetune --model ViT-B/32 --data ./path/to/dataset --output ./path/to/save/model
   ```

   Explanation:
   - `--task finetune`: Indicates that the task is fine-tuning.
   - `--model ViT-B/32`: Specifies the CLIP model to use.
   - `--data ./path/to/dataset`: Path to the training dataset.
   - `--output ./path/to/save/model`: Directory to save the output model.

2. **Evaluate a Pre-Trained Model:**
   Perform inference on a given dataset using a pre-trained model:

   ```bash
   python main.py --task evaluate --model ViT-B/32 --data ./path/to/eval/dataset --pretrained ./path/to/model.pt
   ```

   Explanation:
   - `--task evaluate`: Specifies the evaluation task.
   - `--pretrained ./path/to/model.pt`: Path to the pre-trained model file to evaluate.

3. **Perform Inference:**
   Run inference on specific image and text inputs:

   ```bash
   python main.py --task inference --image ./path/to/image.jpg --text "A description of the image"
   ```

   Explanation:
   - `--image ./path/to/image.jpg`: Path to the input image.
   - `--text "A description of the image"`: Text description to compare with the image.

---

### General Parameters:

| Parameter      | Description                                   | Default            |
|----------------|-----------------------------------------------|--------------------|
| `--task`       | The task to perform (`finetune`, `evaluate`, `inference`). | *required*         |
| `--model`      | The CLIP model to use (`RN50`, `ViT-B/32`, etc.). | `ViT-B/32`         |
| `--data`       | Path to the dataset for fine-tuning or evaluation. | None               |
| `--output`     | Directory to save output models or logs.       | `./output/`        |
| `--image`      | Path to the input image for inference.         | None               |
| `--text`       | Text input for inference tasks.               | None               |
| `--pretrained` | Path to a pre-trained model file.             | None               |

### Notes:

- Ensure that your dataset is prepared and formatted correctly for the respective tasks.
- Fine-tuning may require access to a GPU for optimal performance.
- For more help, run:
  ```bash
  python main.py --help
  ```

---

**Updates will be added as the repository evolves. Feel free to submit an issue or pull request if you encounter any problems or have feature suggestions!**

