# ADS509-module7

### **Title**:  
**Automated Topic Classification and Discovery in News Articles Using Text Mining Techniques**

### **Short Description**:  
This project aims to explore the automated classification and topic discovery of news articles gathered from various categories using the NewsAPI. The primary goal is to build a classification model to categorize articles into predefined groups such as "Technology," "Sports," "Health," and "Business," followed by topic modeling to uncover latent themes. The project will compare how well the discovered topics match with the pre-existing categories.

### **Objectives**:
1. To collect raw news article data via the NewsAPI.
2. To clean, tokenize, and normalize the data for analysis.
3. To build a classification model to categorize the articles based on predefined categories.
4. To build a topic model to discover underlying themes in the data.
5. To evaluate the accuracy of the classification model and compare the topics discovered through topic modeling with the pre-existing categories.
6. To provide insights into how well the discovered topics align with the original article categories.

### **Description of Dataset**:
- **Data Source**: NewsAPI (https://newsapi.org/)
- **Number of Variables**:
  - Each article will include fields such as:
    - `title` (headline of the article)
    - `description` (short summary of the article)
    - `content` (full article text)
    - `source` (publisher of the article)
    - `category` (e.g., technology, sports, health)
    - `publishedAt` (date and time of publication)
- **Size of Dataset**:
  - The dataset size will vary based on the number of articles retrieved per category. For instance, querying 100 articles per category (e.g., "Technology," "Sports," "Health," "Business," etc.) can result in a dataset of approximately 500+ articles.
  - The article text length will also affect the overall size, with typical articles containing 300–1000 words each.

---

Feel free to adjust this based on your specific preferences or any additional data you might gather. Would you like any help refining these sections or further elaborating on the objectives?
