# News Article Topic Classification and Analysis

## Project Description
This project focuses on the automated classification and topic discovery of news articles related to "Trump," gathered using the NewsAPI. The articles were collected over a specified time period and cover various aspects of political, social, and media events. The project employs text mining and natural language processing (NLP) techniques to group the articles into coherent topics and analyze the underlying themes.

## Objective
The primary objectives of this project are:
1. **Classify** news articles into predefined topic groups based on their content.
2. **Discover underlying topics** using Non-negative Matrix Factorization (NMF) and bigram-based text analysis.
3. **Evaluate** the coherence of discovered topics and analyze the distribution of articles across topics.
4. **Provide insights** into the most common topics and their key themes, offering a deeper understanding of the dataset.

## Methodology
- **Data Preprocessing**: Text cleaning and transformation using TF-IDF vectorization with bigrams.
- **Topic Modeling**: NMF was applied to uncover key topics across the dataset, followed by coherence score evaluation to measure topic consistency.
- **Classification**: Each article was assigned to a topic based on the highest contribution from the NMF model.
- **Analysis**: The distribution of topics across articles was visualized, and detailed analysis was conducted on the most common topic.

## Results
- Articles were classified into 5 distinct topics.
- A coherence score of **0.74** was achieved, indicating good semantic consistency in the discovered topics.
- The most common topic included 43 articles, with themes centered around political controversies and media coverage related to Donald Trump.

## Installation and Setup
To reproduce the analysis, follow these steps:

### Requirements
Ensure you have the following Python packages installed:
- `pandas`
- `scikit-learn`
- `matplotlib`
- `gensim`
- `numpy`

You can install these dependencies using the following command:
```bash
pip install pandas scikit-learn matplotlib gensim numpy
```

### Steps
1. **Clone the repository** (if applicable) or download the project files.
2. Place the **news_articles.csv** file in the project directory.
3. Run the Python script for preprocessing, topic modeling, and classification.

## Files
- **news_articles.csv**: Contains the dataset of news articles with columns such as title, description, content, and publication date.
- **topic_classification_script.py**: Main script for text preprocessing, topic discovery using NMF, and classification.
- **README.md**: This file provides an overview of the project, setup instructions, and methodology.

## How to Run the Script
1. Open a terminal or command prompt.
2. Navigate to the directory containing the script and CSV file.
3. Run the script:
   ```bash
   python topic_classification_script.py
   ```

## Additional Notes
- The dataset was preprocessed using bigrams to capture more contextually relevant phrases.
- NMF (Non-negative Matrix Factorization) was chosen for topic modeling due to its ability to generate distinct and interpretable topics.
- The coherence score was calculated using the Gensim library, which evaluates the consistency of words in the topics.
