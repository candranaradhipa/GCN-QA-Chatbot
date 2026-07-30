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
- **Dataset Loading**
The dataset is loaded from a Microsoft Excel `.xlsx` file using the Pandas library. The dataset consists of two primary attributes, namely **Question** and **Answer**, where each row represents a question–answer pair that serves as the chatbot's knowledge base.
- **Tokenization**
Each question and answer is tokenized using the `word_tokenize()` function provided by the Natural Language Toolkit (NLTK). This process splits each sentence into individual tokens, allowing the textual data to be processed more effectively in subsequent stages.
- **Vocabulary Construction**
All tokens extracted from the dataset are combined to create a vocabulary containing unique words. Each word is assigned a unique numerical index, enabling textual data to be represented in a numerical format suitable for machine learning models.
- **Node Feature Representation**
Every token in the vocabulary is transformed into a numerical feature vector using the **One-Hot Encoding** technique. These vectors are utilized as node features, providing numerical representations of words for the Graph Convolutional Network (GCN).
- **Graph Construction**
Each question–answer pair is converted into a graph structure, where every token represents a node and sequential relationships between tokens are represented as edges. Additionally, the last token of the question is connected to the first token of the corresponding answer, enabling the GCN to learn relationships between questions and their associated answers.
- **Graph Dataset Generation**
The constructed graphs are stored as `Data` objects provided by the PyTorch Geometric library. Each graph contains node features `(x)`, edge connections `(edge_index)`, and class labels `(y)`, forming the graph-based dataset used for model training and evaluation.
