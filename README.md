# sentiment_analysis_MLops_project 

## Overview:
This project aims to classify comments into Positive, negative sentiment. For this purpose this project uses sentiment analysis and its related technologies. This project uses multinominal naive bayes algorithm for classification prediction.

## Folder Structure:
Here's an overview of the project's folder structure:
```
Sentiment-Analysis-Project/
├── artifacts/
│ (Include trained models or other important artifacts)
│
├── notebooks/
│ ├── notebook.ipynb (Jupyter Notebook for the project)
│ ├── data/ (Data files or datasets)
│
├── logs/ (Log files or logs directory)
│
├── src/
│ ├── logger.py (Logging utilities)
│ ├── exception.py (Custom exception handling)
│ ├── utils.py (Utility functions)
│ ├── init.py
│
│ ├── components/
│ │ ├── init.py
│ │ ├── data_ingestion.py (Data loading functions)
│ │ ├── data_transformation.py (Data preprocessing and feature engineering)
│ │ ├── model_training.py (Machine learning model training)
│
│ ├── pipelines/
│ │ ├── init.py
│ │ ├── prediction_pipeline.py (Prediction pipeline)
│ │ ├── training_pipeline.py (Model training pipeline)
│
├── requirements.txt (List of project dependencies)
├── nltk.txt
├── setup.py (Setup script for the project)
├── app.py (Main application script)
├── .gitignore (Specify files to ignore in version control)
```

## Project Workflow:
1. **Data Ingestion and Preprocessing**:
- Data is ingested using functions in data_ingestion.py.
- Data preprocessing and feature engineering are performed in data_transformation.py.
2. **Model Training**:
- Machine learning models are trained using the data in model_training.py.
- You can customize the models and hyperparameters as needed.
3. **Prediction Pipeline**:
- The prediction_pipeline.py script handles sentiment classification based on user input.
4. **Notebook**:
- The notebook.ipynb contains an interactive Jupyter Notebook that provides insights into the project.

## Technologies used:
- Python language.
- Libraries used : Streamlit, Pandas, Numpy, Sklearn, Seaborn, etc.
- IDE: VS-code
  
## Execution:
To make use of this project:

- Install the required dependencies listed in requirements.txt.
- Navigate to the notebooks/ directory and open notebook.ipynb to explore and experiment with the project interactively.
- Use the scripts in the src/ directory to execute various components of the project.
- Customize and extend the project as needed for your specific use case.

## Testing:
To run the app follow thin link: https://sentimentanalysismlopsproject-k883l7cibxfgaxpdus3ye5.streamlit.app/

## Author:
- Yash Keshari
- Socials: https://www.linkedin.com/in/yash907/
