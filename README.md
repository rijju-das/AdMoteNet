
# AdMoteNet - Adaptive Multimodal Network for Understanding  Image with Emotional Contexts

A PyTorch‐based multimodal model that classifies sentiments in billboard images by combining visual features (Vision Transformer) with textual features (BERT on extracted captions & object lists).


---

## Setup & Installation

1. **Clone** this repo:
   ```bash
   git clone https://github.com/yourusername/LLMBillboardSentiment.git
   cd LLMBillboardSentiment
   ```

2. **Create & activate** a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data Preprocessing

1. **Image transforms** (in `train.py` / `eval.py`):

   ```python
   transforms.Resize((224,224))
   transforms.ToTensor()
   transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
   ```

2. **Text & Object Features**

   * JSON files in
     `data/features/text-extract/` and
     `data/features/object-extract/`
     mapping each image to:

     * OCR’d caption tokens (list of words)
     * Detected object labels (list of strings)

3. **Sentiment Labels**

   * `data/features/sentiments/Sentiments.json`: maps each image filename to nested lists of sentiment‐ID strings.
   * `data/features/sentiments/Sentiments_List_updated.txt`: human‐readable mapping from ID → “Sentiment Name”.


## Model Architecture

* **Vision Encoder**

  * `timm.create_model('vit_base_patch16_224', pretrained=True)`
  * Remove the classification head (`head = Identity`), output 768-d feature.

* **Text Encoder**

  * `BertModel.from_pretrained('bert-base-uncased')`
  * Tokenize captions & object lists separately → pooler\_output (768-d each).

* **Fusion & Classifier** (`MultimodalModel` in `model.py`):

  ```python
  class MultimodalModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.classifier = nn.Sequential(
              nn.Linear(768*3, 512),
              nn.ReLU(),
              nn.Linear(512, 30)
          )
      def forward(self, img_emb, txt_emb, obj_emb):
          x = torch.cat([img_emb, txt_emb, obj_emb], dim=1)
          return self.classifier(x)
  ```

  * Input: concatenated 2304-d vector
  * Output: 30 raw logits → `BCEWithLogitsLoss` + sigmoid

---

## Training

```bash
python code/train.py \
  --data_dir data/dataset_train_test/train \
  --text_dir data/features/text-extract \
  --obj_dir data/features/object-extract \
  --sentiment_json data/features/sentiments/Sentiments.json \
  --sentiment_list data/features/sentiments/Sentiments_List_updated.txt \
  --output_model code/model/multimodal_model_senti.pt \
  --batch_size 8 \
  --lr 2e-5 \
  --epochs 50
```

Key flags:

* `--batch_size`: images per batch
* `--lr`: learning rate for AdamW
* `--epochs`: number of training epochs

---

## Evaluation

```bash
python code/eval.py \
  --model_path code/model/multimodal_model_senti.pt \
  --data_dir data/dataset_train_test/test \
  --text_dir data/features/text-extract \
  --obj_dir data/features/object-extract \
  --output_csv output/sentiment_predictions__multilabeltestdataset.csv
```

Generates:

* **CSV** with columns:
  `image, true_sentiment_ids, true_sentiment_names, predicted_sentiments, predicted_names, confidence_scores`

* **Metrics** printed to console:

  * Average test loss
  * Average per-sample accuracy (exact‐match fraction)

---

## Visualization

Use `code/visualize.py` to produce:

* **ROC curves** (micro‐avg & per‐class)
* **Confusion matrix grid** for each class
* **Per‐class F1 bar charts** with macro/micro reference lines
* **Training curves** (loss, accuracy vs. epoch) from your `.out` log

Example:

```bash
python code/visualize.py \
  --pred_csv output/sentiment_predictions__multilabeltestdataset.csv \
  --sentiment_list data/features/sentiments/Sentiments_List_updated.txt \
  --log_file train_logs/trainepochoutput.out
```

---

## Citation

This code is released under the MIT License.
If you use it in your research, please cite:

> **Your Name**, “Multimodal Billboard Sentiment Classification with ViT & BERT,” *GitHub*, 2025.
> [https://github.com/yourusername/LLMBillboardSentiment](https://github.com/yourusername/LLMBillboardSentiment)

