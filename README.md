### **Title**:  
**Automated Topic Classification and Discovery in News Articles Related to Trump Using Text Mining Techniques**

### **Short Description**:  
This project focuses on automating the classification and topic discovery of news articles related to "Trump" gathered using the NewsAPI. Articles were collected over a specific period and cover various aspects of the topic. The aim is to classify these articles into predefined groups and explore underlying topics through text mining techniques, evaluating how well the discovered topics align with the predefined categories.

### **Objectives**:
1. To collect raw news article data using the NewsAPI for articles about Trump.
2. To clean, tokenize, and normalize the dataset for analysis.
3. To build a classification model to categorize the articles based on their source or theme.
4. To build a topic model to discover underlying topics in the articles.
5. To evaluate the classification model’s accuracy and compare the topics discovered via topic modeling with the original article themes.
6. To provide insights on how the discovered topics align with article sources or categories.

### **Description of Dataset**:
- **Data Source**: NewsAPI (https://newsapi.org/)
- **API Query**: 
  - Query: `"Trump"`
  - Date Range: 2024-09-05 to 2024-10-01
- **Number of Variables**:
  - Each news article contains the following fields:
    - `source` (news source of the article)
    - `author` (author of the article)
    - `title` (headline of the article)
    - `description` (brief summary of the article)
    - `content` (full text of the article)
    - `publishedAt` (publication date and time)
    - `url` (link to the article)
    - `urlToImage` (link to the article's image)
- **Number of Records**:
  - The dataset size will vary based on the number of articles matching the query within the date range, but it could range from several dozen to a few hundred articles.
- **Size of Dataset**:
  - The articles range in length from 300 to 1000 words each.
- **Variables for Analysis**:
  - The primary text fields for analysis include `title`, `description`, and `content`, which will be used in both classification and topic modeling tasks.
