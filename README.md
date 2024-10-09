# Next-Word-Prediction
Implemented Next-Word Prediction using MLP, LSTM, and Transformer-based Language Models

## **Overview**

This assignment implements three different types of language models: 
1. **Neural Network Language Model (MLP)** based on a 6-gram context.
2. **LSTM-based Language Model**.
3. **Transformer Decoder-based Language Model**.

Each of these models is trained on the **Auguste Maquet** dataset, and their performances are evaluated using **perplexity** scores.

The assignment involves tokenizing the input text, using pre-trained GloVe embeddings, and training the models to predict the next word in a sequence. For each model, both validation and test perplexities are calculated and stored in separate files.

---
## Assumptions:-

- For N-gram, the value of N is taken as 6 as the assignment pdf stated to take 5 words for context and 1 for focus word.
- The Hyperparameter Tuning of Neural Network Language Model is done for only 10 epochs while the actual model was trained for 15 due to time and processing constraints.

---

## **Requirements**

- **Python 3.x**
- **PyTorch**
- **NumPy**
- **NLTK**
- **Scikit-learn**
- Pre-trained **GloVe embeddings** (300-dimensional)

Ensure that the required libraries are installed. You can install them using the following command:

```bash
pip install torch numpy nltk scikit-learn
```
## How To Execute
### **Step 1: Prepare the Dataset**
The dataset `Auguste_Maquet.txt` should be placed in the working directory or specify the correct path in the code.
Preprocess the dataset by tokenizing sentences and words, followed by cleaning (removing special characters, converting to lowercase, etc.).

### **Step 2: Load Pre-trained GloVe Embeddings**
The GloVe embeddings file (`glove.6B.300d.txt`) is required for the project. The code automatically loads and maps the embeddings based on the vocabulary from the dataset.

Ensure that the path to the GloVe embeddings is correct in the code.

### **Step 3: Train the Models**
To train any of the models (MLP, LSTM, Transformer), run the corresponding functions and classes. For each model:
1. The dataset is split into **training**, **validation**, and **test sets**.
2. The model is trained using the training set, and the **validation perplexity** is calculated at each epoch.
3. After training, the model is evaluated on the test set to compute the **test perplexity**.

For each model, the results (validation and test perplexities) are saved in respective text files.

#### **Training the Models**


```bash
python 2023201020_assignment1.py
```
### **Step 4: Evaluation and Results**

The `evaluate` function is used to calculate perplexities on the validation and test sets. The results will be saved in the following format:

- **Validation perplexity file**:
    - Format: `Batch No \t Perplexity`
    - Last line: `Average Perplexity: <value>`
  
- **Test perplexity file**:
    - Format: `Batch No \t Perplexity`
    - Last line: `Average Perplexity: <value>`

The results will help measure the effectiveness of the model on unseen data by analyzing the perplexity values, where lower values represent better performance.

---

## **Restoring Pre-trained Models**

We can restore the pre-trained models using the saved `.pth` files for further testing or inference. The weights of each model can be restored by using the following commands:

```python
# For MLP model
model.load_state_dict(torch.load('neural_language_model.pth'))

# For LSTM model
model.load_state_dict(torch.load('lstm_language_model.pth'))

# For Transformer model
model.load_state_dict(torch.load('transformer_model.pth'))
```

## Hyperparameter Tuning for Neural Language Model(Bonus)
The image of the graph is attached with this readme file. From the hyperparameter tuning analysis we can infer that, setting both the hidden layers of MLP to 250, learning rate=0.003 and optimizer= SGD, we get the best possible results as test and validation perplexities are the least, even when the model is run for 10 epochs.

## Final Results 
- Neural Network Language Model:- 
> Validation Perplexity: 329.4407   
> Test Perplexity: 341.5118
- LSTM Language Model:- 
> Validation Perplexity: 206.2503  
> Test Perplexity: 208.2038
- Transformer Language Model:- 
> Validation Perplexity: 27.0166   
> Test Perplexity: 28.3719

