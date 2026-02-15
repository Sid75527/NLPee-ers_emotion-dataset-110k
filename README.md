# NLPee-ers - Emotion Classification using BERT

## Team Information

**Team Name:** NLPee-ers  

**Members:**
- Shreyas S Mallappa – PES1UG23AM296
- Siddharth M – PES1UG23AM303
- Sanat Shirwaicar – PES1UG23AM266
- Shreyansh Subham – PES1UG23AM294


## Dataset Used
Emotion Dataset (110k)  
Source: HuggingFace - shreyaspullehf/emotion_dataset_100k

---

## Project Overview

This project explores transfer learning by fine-tuning a pre-trained BERT model for multi-class emotion classification using PyTorch.

The model was trained using a custom PyTorch training loop (no HuggingFace Trainer API).

---

## Approach

1. Loaded the dataset using HuggingFace `datasets`.
2. Performed Exploratory Data Analysis (EDA) to examine class distribution and text length.
3. Tokenized the text using `BertTokenizer` with a maximum sequence length of 128.
4. Fine-tuned a pre-trained `bert-base-uncased` model using a custom PyTorch training loop.
5. Evaluated the model using Accuracy, Precision, Recall, F1-score (weighted), and a Confusion Matrix.
6. Implemented an inference pipeline to predict emotion labels for custom input text.

---

## Assumptions

- The dataset is correctly labeled and does not contain significant noise.
- Since the dataset is balanced, no additional class weighting or resampling techniques were applied.
- A maximum sequence length of 128 tokens is sufficient, as most sentences in the dataset are relatively short.
- The task is treated as a single-label classification problem (one dominant emotion per sentence).

---

## Observations

- The dataset is well-balanced across all emotion classes.
- The fine-tuned BERT model converged quickly and achieved strong performance (F1 ≈ 0.95).
- The model performs well on explicit emotional expressions.
- Some limitations were observed in handling negation and mixed-emotion sentences.
- The confusion matrix indicates minor confusion between semantically similar emotions.

---

## Dataset Description

The dataset contains 10 emotion classes:

- drive  
- excitement  
- disgust  
- fear  
- happiness  
- sadness  
- embarrassment  
- love  
- loneliness  
- surprise  

The dataset is well-balanced across all emotion categories.

---

## Exploratory Data Analysis (EDA)

- Class distribution visualized using seaborn
- No significant class imbalance observed
- Text length distribution analyzed to justify tokenization length

---

## Model Architecture

- Model: `bert-base-uncased`
- Tokenizer: `BertTokenizer`
- Classification Head: Linear layer (10 classes)
- Optimizer: AdamW
- Learning Rate: 2e-5
- Batch Size: 32
- Epochs: 2
- Max Sequence Length: 128

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Confusion Matrix (heatmap visualization)

### Final Performance

- Accuracy: ~0.95
- Weighted F1 Score: ~0.95

The model demonstrates strong generalization performance across all emotion classes.

---

## Inference Pipeline

A custom `predict_text()` function was implemented to:
- Accept raw input text
- Return predicted emotion label
- Return confidence score

---

## Limitations

The model shows minor limitations in handling:
- Negation (e.g., "not unhappy")
- Mixed-emotion sentences

This is expected in single-label emotion classification tasks.

---

## How to Run

1. Install dependencies:

