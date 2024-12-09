### **README.md**

# **Neural-based Propaganda Detection**
This repository implements a neural network-based classifier to detect propaganda in textual data. The model is trained using GloVe word embeddings and a Multi-Layer Perceptron (MLP) architecture. It predicts whether a given sentence belongs to the class of *"propaganda"* or *"non-propaganda"*.

---

## **Features**
- Pre-trained GloVe embeddings for semantic word representations.
- Robust neural architecture using PyTorch.
- Pickle-based model serialization for easy inference.
- Simple and efficient sentence vectorization using GloVe.
- CLI-based inference for predicting individual sentences.

---

## **Technologies and Tools**

### **1. Python Libraries**
- **PyTorch:** Deep learning framework used for building and training the MLP classifier.
- **Gensim:** For loading and utilizing pre-trained GloVe word embeddings.
- **NLTK:** Tokenizer for breaking sentences into words.
- **NumPy:** Efficient numerical computations for vector operations.
- **Scikit-learn:** For evaluating the model (precision, recall, F1-score).
- **Pandas:** Data manipulation during preprocessing.

### **2. Pre-trained Embeddings**
- **GloVe (Global Vectors for Word Representation):**
  - Version: `glove.6B.300d.txt`.
  - Provides 300-dimensional dense vector representations for English words.

### **3. Dataset**
- Textual data provided in `train.tsv`.
  - Contains sentences, article titles, and their corresponding labels (`propaganda` or `non-propaganda`).

### **4. File Serialization**
- **Pickle:**
  - Saves the trained model weights, input parameters, and OOV vector for efficient inference.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/ash-sha/propaganda-detection.git
cd propaganda-detection
```

### **2. Install Dependencies**
Ensure Python 3.8+ is installed. Then install required libraries:
```bash
pip install -r requirements.txt
```

### **3. Download Pre-trained GloVe Embeddings**
- Download `glove.6B.300d.txt` from the [GloVe website](https://nlp.stanford.edu/projects/glove/).
- Place the file in the appropriate directory (e.g., `./glove.6B.300d.txt`).

### **4. Training**
(Optional) To retrain the model from scratch:
```bash
train.ipynb
```

### **5. Inference**
- Ensure the model file (`propoganda.pickle`) is present in the working directory.
- Run the inference script:
```bash
test.ipynb
```
- Enter your sentence when prompted, and get the prediction:
```plaintext
Enter the query: The government is spreading fake news to mislead the public.
Predicted label: propaganda
```

---

## **Repository Structure**
```plaintext
propaganda-detection/
│
├── glove.6B.300d.txt   # Pre-trained GloVe embeddings
├── train.tsv   # Dataset used for training
├── propoganda.pickle   # Serialized trained model
├── train.ipynb                # Script for training the model
├── test.ipynb            # Script for running inference
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

---

## **Model Workflow**

1. **Data Preprocessing:**
   - Load sentences and labels from the dataset.
   - Shuffle and split data into training, validation, and testing sets.
   - Normalize and vectorize sentences using GloVe embeddings.

2. **Model Architecture:**
   - Multi-Layer Perceptron (MLP) with one hidden layer.
   - Dropout regularization to prevent overfitting.

3. **Training:**
   - Optimize using the Adam optimizer and CrossEntropy loss.
   - Evaluate on the validation set after each epoch.
   - Save the model with the highest validation F1-score.

4. **Inference:**
   - Load the trained model (`propoganda.pickle`).
   - Use GloVe embeddings to vectorize the input query.
   - Predict the class label for the input sentence.

---

## **Sample Results**
- **Input Query:** `The government is spreading fake news to mislead the public.`
- **Predicted Label:** `propaganda`

---

## **Future Work**
- Add support for multi-class propaganda classification (e.g., detecting specific propaganda techniques).
- Improve sentence vectorization by integrating contextual embeddings (e.g., BERT, RoBERTa).
- Implement a web or mobile interface for real-time predictions.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.