# Question-Answering (QA) Chatbot Using Graph Convolutional Network

## Overview
This project develops a Question Answering (QA) chatbot that provides information about the Faculty of Advanced Technology and Multidiscipline, Universitas Airlangga. The system integrates web scraping for data acquisition with a Graph Convolutional Network (GCN) to process and retrieve relevant information in response to user queries.

## Methodology
### Dataset
The dataset used in this project was collected through a web scraping process from the official website of the Faculty of Advanced Technology and Multidiscipline (FTMM), Universitas Airlangga. The collected data includes various types of information, such as faculty member profiles from each study program, the dean's profile, student communities and organizations under the FTMM faculty, as well as other relevant institutional information. The scraped data were then processed and stored in a Microsoft Excel (.xlsx) file containing a total of 53 information entries, which serve as the primary knowledge base for the system.

### Tools
- Python 3.11.9
- Jupyter Notebook
- Visual Studio Code (optional)
- Python Libraries (nltk, torch, pandas, scikit-learn)

### Data Preprocessing
- **Dataset Loading** :
The dataset is loaded from a Microsoft Excel `.xlsx` file using the Pandas library. The dataset consists of two primary attributes, namely **Question** and **Answer**, where each row represents a question–answer pair that serves as the chatbot's knowledge base.
- **Tokenization** :
Each question and answer is tokenized using the `word_tokenize()` function provided by the Natural Language Toolkit (NLTK). This process splits each sentence into individual tokens, allowing the textual data to be processed more effectively in subsequent stages.
- **Vocabulary Construction** :
All tokens extracted from the dataset are combined to create a vocabulary containing unique words. Each word is assigned a unique numerical index, enabling textual data to be represented in a numerical format suitable for machine learning models.
- **Node Feature Representation** :
Every token in the vocabulary is transformed into a numerical feature vector using the **One-Hot Encoding** technique. These vectors are utilized as node features, providing numerical representations of words for the Graph Convolutional Network (GCN).
- **Graph Construction** :
Each question–answer pair is converted into a graph structure, where every token represents a node and sequential relationships between tokens are represented as edges. Additionally, the last token of the question is connected to the first token of the corresponding answer, enabling the GCN to learn relationships between questions and their associated answers.
- **Graph Dataset Generation** :
The constructed graphs are stored as `Data` objects provided by the PyTorch Geometric library. Each graph contains node features `(x)`, edge connections `(edge_index)`, and class labels `(y)`, forming the graph-based dataset used for model training and evaluation.

### GCN Model Architecture
The Graph Convolutional Network (GCN) architecture implemented in this study consists of two Graph Convolutional (GCNConv) layers designed to learn feature representations from the constructed graph structure. In the first layer, the initial node features are processed using GCNConv(input_dim, hidden_dim) to generate richer feature representations by aggregating information from neighboring nodes. The resulting features are then passed through the Rectified Linear Unit (ReLU) activation function to introduce non-linearity, enabling the model to capture more complex patterns within the graph. Subsequently, the transformed features are processed by a second GCNConv layer (GCNConv(hidden_dim, output_dim)) to produce the final feature representations for classification. Before generating the final prediction, Global Mean Pooling (torch.mean()) is applied to aggregate the features of all nodes into a single graph-level representation by computing their average. This graph representation is then used as the input to the classification stage to predict the most appropriate answer corresponding to the user's query.

> #### GCN Model Workflow <a name = 'workflow'></a>
The Graph Convolutional Network (GCN) model is trained using the graph dataset generated during the preprocessing stage. During each training epoch, every graph is passed through two Graph Convolutional (GCNConv) layers followed by a ReLU activation function and a Global Mean Pooling layer to obtain a graph-level representation. The prediction error is computed using the CrossEntropyLoss function, while the model parameters are optimized using the Adam optimizer through backpropagation. This training process is repeated for **10 epochs** to enable the model to learn the relationships between questions and their corresponding answers.

```text
Graph Dataset
      │
      ▼
Forward Propagation
      │
      ▼
GCNConv Layer 1
      │
      ▼
ReLU Activation
      │
      ▼
GCNConv Layer 2
      │
      ▼
Global Mean Pooling
      │
      ▼
Prediction
      │
      ▼
CrossEntropy Loss
      │
      ▼
Backpropagation
      │
      ▼
Adam Optimizer
      │
      ▼
Weight Update
      │
      ▼
Repeat for 10 Epochs
```

> #### Configuration <a name = 'config'></a>

| Parameter | Value |
|:----------|:------|
| **Model** | Graph Convolutional Network (GCN) |
| **Graph Convolution Layers** | 2 |
| **Activation Function** | ReLU |
| **Loss Function** | CrossEntropyLoss |
| **Optimizer** | Adam |
| **Epochs** | 10 |

The objective of the training process is to minimize the classification error by learning the structural relationships between question and answer graphs. The optimized model is then used to predict the most relevant answer for a given user query.

### Training and Testing
To manage and evaluate the dataset, this study employs the K-Fold Cross Validation method, which systematically partitions the dataset into multiple subsets. This approach divides the dataset into K equally sized folds. During each iteration, K−1 folds are used as the training set, while the remaining one fold serves as the testing set. The process is repeated K times, allowing each fold to be used as the testing set exactly once. In this implementation, the configuration `kfold(n_splits=5, shuffle=True)` is adopted, indicating that the dataset is divided into five folds. Consequently, each iteration uses four folds (80%) for training and one fold (20%) for testing. This approach provides a more reliable evaluation of the model, reduces bias caused by a single train–test split, and offers a more representative estimate of the model's generalization performance.

The following figure illustrates the workflow of the K-Fold Cross Validation process.
<img width="885" height="613" alt="image" src="https://github.com/user-attachments/assets/6d31124f-795c-40fb-8c0e-7467a2e23bf9" />

Source : http://scikit-learn.org/stable/modules/cross_validation.html

The model was trained for 10 epochs in each fold of the K-Fold Cross Validation scheme. During the training process, the model computed the training loss and training accuracy at every epoch to monitor the learning progress and evaluate its performance on the training data. After completing the training phase for a given fold, the model was evaluated using the corresponding testing set. The evaluation results were reported using several performance metrics, including test loss, test accuracy, test precision, test recall, and test F1-score, which were used to assess the model's classification performance on previously unseen data.

After completing the training process for all **10 epochs** across the **5-fold cross-validation** procedure, the model performance is evaluated by averaging the results obtained from each fold. The evaluation is conducted using four standard classification metrics: **Accuracy**, **Precision**, **Recall**, and **F1-Score**. These metrics provide a comprehensive assessment of the model's ability to correctly classify question–answer pairs and measure its overall predictive performance.

The evaluation metrics are calculated using the following formulas:
**Accuracy**

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

**Macro Precision**

$$
Precision_{macro}=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{TP_i}{TP_i+FP_i}
$$

**Macro Recall**

$$
Recall_{macro}=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{TP_i}{TP_i+FN_i}
$$

**Macro F1-Score**

$$
F1_{macro}=
\frac{1}{N}
\sum_{i=1}^{N}
2
\cdot
\frac{Precision_i \times Recall_i}
{Precision_i+Recall_i}
$$

## Results

The following section presents the training and testing results obtained from the first and the final folds as representative examples of the K-Fold Cross Validation process.
<img width="1788" height="407" alt="image" src="https://github.com/user-attachments/assets/660f832a-2ade-4104-bd73-b6badb8775cd" />
<img width="1548" height="348" alt="image" src="https://github.com/user-attachments/assets/bc93d6c7-5265-4558-91dc-162e5d7598f4" />

The following table provides a detailed summary of the training and evaluation results for each fold, including the performance at every epoch.

| Fold | Epoch | Final Training Loss | Final Training Accuracy | Test Loss | Test Accuracy | Precision | Recall | F1-Score |
|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 0.1920 | 100.00% | 12.8391 | 0.00% | 0.0000 | 0.0000 | 0.0000 |
| 2 | 10 | 0.0609 | 100.00% | 0.4480 | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| 3 | 10 | 0.0079 | 100.00% | 0.0836 | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| 4 | 10 | 0.0043 | 100.00% | 0.0058 | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| 5 | 10 | 0.0027 | 100.00% | 0.0019 | 100.00% | 1.0000 | 1.0000 | 1.0000 |

The evaluation results are presented in the following table.
| Evaluation Metric | Average Value |
|:------------------|--------------:|
| **Test Accuracy** | **80.00%** |
| **Test Loss** | **2.6757** |
| **Test Precision** | **0.8000** |
| **Test Recall** | **0.8000** |
| **Test F1-Score** | **0.8000** |


## Model Usage
Once the model has been trained, the system proceeds to the inference stage by accepting a user query as input. The query is first preprocessed and transformed into a graph representation suitable for the Graph Convolutional Network (GCN) model. The trained model then analyzes the input and predicts the most relevant answer based on the knowledge acquired during training. Finally, the predicted answer is mapped to the corresponding response and displayed to the user.

```
Search Here :jelaskan informasi mengenai bu Amilah!
Predicted Answer: NIK : 199210262020013201\nNama : Amila Sofiah, S.T., M.T.\nPendidikan : S2 Teknik Elektro, Institut Teknologi Bandung\nResearch Interest : Biomedical Signal Processing, Instrumentation & Control, Robotics\nEmail : amila.sofiah@ftmm.unair.ac.id
```



