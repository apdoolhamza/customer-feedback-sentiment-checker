# Customer Feedback Sentiment Analysis (ML + Flask) 🧠

A simple yet powerful sentiment analysis system that predicts whether customer feedback is **Positive** or **Negative** with a confidence percentage using **python** and **natural language processing (NLP) techniques**, this project enables businesses to gain insights into customer opinions, improving decision-making and customer experience.

[![Watch the video](https://img.youtube.com/vi/ULMRtAO1rqA/maxresdefault.jpg)](https://youtu.be/ULMRtAO1rqA)

## Features

- **Sentiment Classification**: Accurately categorizes customer feedback into positive or negative sentiments.
- **Text Preprocessing**: Cleans and processes feedback data with tokenization, stop-word removal, and vectorization.
- **Web Interface**: User-friendly interface for inputting feedback and viewing sentiment results.

## Technologies Used
- **Python**: Core language for data processing and machine learning.
- **Scikit-learn**: For building and evaluating machine learning models.
- **NLTK**: For NLP tasks like sentiment analysis and text preprocessing.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Flask, HTML5, & Bootstrap 5**: For deploying the model as a web application.
- **Jupyter Notebook**: For exploratory data analysis and model development.

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/apdoolhamza/customer-feedback-sentiment-checker.git
```
### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the project
```bash
python app.py
```
## Usage
1. Launch the application using the instructions above.
2. Input customer feedback (e.g., “The service was excellent!”) into the interface.
3. Receive a sentiment prediction (e.g., Positive or Negative).
   
## Dataset
The model was trained on a simplified version of the Amazon Cell Phones Reviews Dataset.

## Possible Improvements
* REST API version (/api/sentiment)
* Save feedback + prediction in DB
* Admin dashboard with Chart.js
* Multi-class support (neutral, mixed)
* HuggingFace/BERT upgrade

## Contact
Apdoolmajeed Hamza - [LinkedIn Profile](https://www.linkedin.com/in/apdoolhamza/) | [GitHub Profile](https://github.com/apdoolhamza/)

##  License
This project is licensed under the [MIT License](https://github.com/apdoolhamza/customer-feedback-sentiment-checker/blob/main/LICENSE)
