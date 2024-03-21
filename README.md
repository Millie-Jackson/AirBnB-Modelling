
# Badges
[![Build Status](https://travis-ci.org/Millie-Jackson/airbnb-data-analysis.svg?branch=main)](https://travis-ci.org/Millie-Jackson/airbnb-data-analysis)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9-blue)](https://www.python.org/downloads/)

# AirBnB
Data Science Specialization Project With AiCore
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team

This project involves cleaning and analyzing Airbnb data to gain insights and understand trends in the rental market. 
The dataset used in this project contains various features related to Airbnb listings, such as property details, amenities, ratings, and pricing.



# Table of Contents
1. [Introduction](#introduction)
    - [Problem](#problem)
    - [Solution](#solution)
    - [Target Users](#target-users)
    - [Motivation](#motivation)
2. [Major Features](#major-features)
    - [Data Cleaning](#data-cleaning)
    - [Real-World Learning Exercise](#real-world-learning-exercise)
    - [Customizable Label Column](#customizable-label-column)
    - [Data Saving](#data-saving)
    - [User-Friendly Execution](#user-friendly-execution)
    - [Modular Code Structure](#modular-code-structure)
    - [Real-World Dataset - Airbnb](#real-world-dataset---airbnb)
    - [What Makes this Project Unique](#what-makes-this-project-unique)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [File Structure](#file-structure)
7. [License](#license)
8. [Technical Breakdown](#technical-breakdown)
    - [Data Loading](#data-loading)
    - [Data Cleaning](#data-cleaning-1)
    - [Save Cleaned Data](#save-cleaned-data)
    - [Label Selection](#label-selection)
    - [Learnings](#learnings)



## Introduction
Airbnb is a popular online marketplace for renting vacation homes, apartments, and lodging. 
The data collected from Airbnb listings can provide valuable information about rental trends, guest preferences, and host performances. 
This project aims to clean the raw Airbnb data, handle missing values, and perform data analysis to extract meaningful insights.
Overall, the project's aim is to empower users with meaningful information and trends in the vacation rental market, making it a valuable resource for various stakeholders interested in the Airbnb industry.

### Problem
The Airbnb Data Cleaning and Analysis Project aims to clean and analyze raw Airbnb data to provide valuable insights and trends in the vacation rental market. 
The dataset contains diverse information related to Airbnb listings, including property details, amenities, guest ratings, and pricing. 
By addressing missing data, handling inconsistencies, and performing data analysis, this project seeks to transform the raw data into a structured and meaningful format.

Specifically, we want to address the following:

- How can we handle missing data and inconsistencies in the raw Airbnb dataset?
- How can we extract meaningful insights and patterns from the cleaned dataset?
- How can we identify trends in the vacation rental market, such as pricing patterns, amenities preferences, and guest ratings?
- How can we provide a user-friendly solution for exploring and visualizing the data?

### Solution
Through the Airbnb Project, users gain access to a cleaned and structured dataset with valuable insights into the vacation rental market. By addressing missing data, performing thorough analysis the project allows property owners, travelers, investors, and data enthusiasts to explore trends, make informed decisions and gain a deeper understanding of the Airbnb market. The user-friendly implementation and data exploration capabilities make the project a valuable resource for anyone interested in the vacation rental market. 

Whether you are looking to optimize your rental offerings, plan your travels, or analyze investment opportunities, this project equips you with the tools to gain actionable insights from Airbnb data.

### Motivation
The Airbnb Data Cleaning and Analysis Project was undertaken as part of the AiCore Data Science Specialization. It served as my first attempt at building a prediction model using neural networks and was a valuable learning exercise utilizing a real-world example.

As part of the data science specialization, the project aimed to enhance my understanding of data cleaning, data analysis, and the application of neural networks for predictive modeling. By working on a real-world dataset like Airbnb data, I could explore the intricacies of handling missing data, extracting insights, and identifying trends in the vacation rental market.

Throughout the project, I gained hands-on experience in data preprocessing, data visualization, and implementing neural networks for regression tasks. The aim was to develop a comprehensive solution that could be applied to other similar datasets and provide meaningful insights to diverse stakeholders in the vacation rental market.

The project's motivation was not only to build a prediction model but also to grasp the practical challenges faced while working with real data. It allowed me to refine my data science skills, learn from iterative improvements, and enhance my overall understanding of the data science workflow.

### Target Users
The target users of this project are individuals or organizations interested in gaining a deeper understanding of the Airbnb rental market. 
Potential users include:

**Property Owners and Hosts:** Hosts looking to optimize their rental offerings by understanding pricing trends, amenities preferences, and guest ratings.

**Travelers and Guests:** Travelers seeking valuable insights into popular locations, amenities availability, and the overall quality of accommodations.

**Real Estate Investors:** Investors exploring potential investment opportunities in the vacation rental market.

**Data Enthusiasts:** Individuals interested in data analysis, visualization, and gaining insights from real-world datasets.

## Major Features
**Data Cleaning:** The project includes a robust data cleaning process to handle missing values, data type conversion, and inconsistency removal. It ensures a clean and reliable dataset for analysis.

**Regression Modelling:** Tur regression modeling is tailored for predicting continuous numeric outcomes, such as pricing in the context of Airbnb listings. Leveraging advanced machine learning techniques, we utilize algorithms like Linear Regression, Decision Trees, Random Forests, and Gradient Boosting to uncover intricate patterns within the dataset. The pipeline involves meticulous data preprocessing, feature engineering, and hyperparameter tuning to ensure optimal performance. Users can easily experiment with different regression models by following a standardized process that involves model training, evaluation, and saving for future use.

**Classification Modelling:** The system excels at predicting categorical outcomes, a crucial task for tasks like predicting Airbnb listing categories. We employ diverse classification algorithms, including Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting. Through the use of hyperparameter tuning and careful model evaluation, users can identify the best-performing classification model for their specific use case. The provided framework allows for a seamless workflow, from loading and preprocessing data to training, evaluating, and saving classification models for deployment.

**Neural Network Predictive Model:** As a significant feature, the project implements a neural network predictive model for regression tasks. It is my first attempt at building a prediction model using neural networks, providing valuable hands-on experience in machine learning.

**Real-World Learning Exercise:** The project serves as a learning exercise utilizing a real-world example of Airbnb data. It allowed me to grasp practical challenges faced while working with real data and refine data science skills.

**Exploratory Data Analysis:** This program employs a comprehensive data analysis approach, primarily driven by exploratory data analysis (EDA). Leveraging the powerful Pandas library, we conduct in-depth statistical analyses to extract meaningful insights from the dataset.

- **Descriptive Statistics:**
  - Utilizing Pandas, we calculate key descriptive statistics such as mean, median, standard deviation, and quartiles. These measures provide a summary of the central tendency and dispersion of numerical features.

- **Correlation Analysis:**
  - We delve into the relationships between variables by computing correlation coefficients. This allows us to identify patterns and dependencies among different features.

- **Distribution Analysis:**
  - Employing histograms and kernel density plots, we visualize the distribution of numerical variables. This aids in understanding the data's underlying structure and potential skewness.



**Visualization:** The visualization aspect, powered by Matplotlib, plays a pivotal role in transforming intricate patterns into accessible insights. This holistic approach to data analysis and visualization not only enriches data-driven decision-making but also weaves a compelling narrative, facilitating effective communication of our findings.

- **Pairwise Scatter Plots:**
  - We create scatter plots to visualize pairwise relationships between numerical features. This enables us to identify potential patterns, clusters, or outliers.
 
- **Categorical Data Analysis:**
  - For categorical variables, we generate bar charts and frequency tables to explore the distribution of different categories and understand their impact on the target variable.

**Customizable Label Column:** The solution offers the flexibility to specify a target label column during analysis. Users can focus on specific aspects of the rental market by customizing the analysis based on their interests.

**Data Saving:** The cleaned data is saved as "clean_tabular_data.csv" to facilitate future analysis and usage. Additionally, the program systematically preserves the trained model along with its associated hyperparameters and performance metrics. This includes storing the model in a designated folder, such as "models/classification/logistic_regression," and creating separate files for hyperparameters ("hyperparameters.json") and performance metrics. This meticulous approach ensures the reproducibility of results and provides a comprehensive record of the model training process.

**Neural Network Training:** The Neural_Network.py script contains functions for training neural networks on the cleaned tabular data. You can train a model using the train function and specify hyperparameters in a YAML config file.

**User-Friendly Execution:** The project is designed with user-friendliness in mind, providing easy execution of data cleaning and analysis scripts. Individuals with varying programming experience can explore the Airbnb dataset effortlessly.

**Modular Code Structure:** The implementation follows a modular code structure, promoting code reusability and maintainability. Each major feature is encapsulated in separate functions, making it easy to understand and modify the code.

**Real-World Dataset - Airbnb:** The project employs a real-world dataset from Airbnb, ensuring the relevance and practical applicability of the analysis and model building.

### What Makes this Project Unique
The project stands out due to its user-friendly implementation, allowing individuals with varying programming experience to explore the Airbnb dataset effortlessly. The ability to specify a target label column during analysis also provides users with customization options based on their specific interests and requirements.



## Installation
To run the code in this project, you'll need to have Python installed on your system. Additionally, the following Python libraries are required and can be installed using pip:

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- tensorboard

You can install the dependencies using pip:

```bash
pip install -r requirements.txt



## Usage
Clone the repository to your local machine:

```python git clone https://github.com/Millie-Jackson/airbnb-data-analysis.git```

Navigate to the project directory:

```python cd airbnb```

Run the data cleaning and analysis scripts:

```python python tabular_data.py```

Run the script for regression modeling to train and evaluate machine learning models for predicting nightly prices:

```bash python regression_modelling.py```

Run the script for classification modeling to train and evaluate machine learning models for predicting catagories:

```bash python classification_modelling.py```

Train the neural network:

'''python Neural_Network.py'''


## Contributing
Contributions to this project are welcome! 
If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.



## File Structure

AirBnB/
│
├── data/
│   ├── airbnb-property-listings/
│   └── images/
│
├── tabular_data/
│   ├── clean_tabular_data.csv
│   └── listing.csv
│
├── docs/
│   ├── Learning Materials
│   ├── screenshots
│   └── links.txt
│
├── models/
│   ├── classification/
│   │   ├── decisiontreeregressor/
│   │   │   ├── model.joblib
│   │   │   └── hyperparameters.json
│   │   ├── gradientboostingregressor/
│   │   │   ├── model.joblib
│   │   │   └── hyperparameters.json
│   │   ├── logisticregressor/
│   │   │   ├── model.joblib
│   │   │   └── hyperparameters.json
│   │   └── randomforestregressor/
│   │       ├── model.joblib
│   │       └── hyperparameters.json
│   │
│   └── regression/
│       ├── decisiontreeregressor/
│       │   ├── model.joblib
│       │   ├── hyperparameters.json
│       │   └── metrics.json
│       ├── randomforestregressor/
│       │   ├── model.joblib
│       │   ├── hyperparameters.json
│       │   └── metrics.json
│       └── gradientboostingregressor/
│           ├── model.joblib
│           ├── hyperparameters.json
│           └── metrics.json
│
├── src/
│   └── __init__.py
│
├── utils/
│   └── __init__.py
│   ├── classification_model.py
│   ├── data_analysis.py
│   ├── data_cleaner.py
│   ├── Modelling.py
│   ├── Neural_network.py
│   ├── nn_config.yaml
│   └── tabular_data.py
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt

## License
This project is licensed under the MIT License.



# Technical Breakdown
**Data Loading:** The code starts by loading the raw Airbnb data from a CSV file ("listing.csv") using the pandas library's read_csv function. It handles the case when the file is not found and raises a FileNotFoundError with an appropriate error message.

**Data Cleaning:** The data cleaning process involves removing missing values, converting data types, and handling inconsistencies. 

The tabular_data.py script calls the "clean_tabular_data(df)" function to perform data cleaning. It serves as the core data cleaning process. 

Makes a copy of the original DataFrame ("df_before_update") to track changes.

Calls the remove_rows_with_missing_ratings(df), combine_description_strings(df), and set_default_feature_values(df) functions to clean the data.

Compares if the DataFrame ("df") has been modified after the update and displays a message indicating whether the original DataFrame has been updated.

Reindexes the DataFrame and removes the old index.
![Code Screenshot](screenshots/clean_tabular_data.png)

The "remove_rows_with_missing_ratings(df)" function is called to rows with missing values in the rating columns (Accuracy_rating, Communication_rating, Location_rating, Check-in_rating, Value_rating).
![Code Screenshot](screenshots/remove_rows_with_missing_ratings.png)

The "combine_description_strings(df)" function is called to combine and clean the strings in the "Description" column by removing missing descriptions (NaN), the prefix "'About this space'," and empty quotes from the lists in the "Description" column.
![Code Screenshot](screenshots/combine_description_strings.png)

The "set_default_feature_values(df)" function is called to set default values for the feature columns (guests, beds, bathrooms, bedrooms) to fill missing values with 1.
![Code Screenshot](screenshots/set_default_feature_values.png)

**Save Cleaned Data:** After data cleaning, the cleaned DataFrame ("df") is saved as "clean_tabular_data.csv" using the to_csv method of pandas.

**Label Selection:**
The program loads the cleaned data from "clean_tabular_data.csv" using pandas to proceed with further analysis.

The program calls the "load_airbnb(label='Price_Night')" function to load the cleaned data for analysis, with "Price_Night" as the label column.

The "load_airbnb" function checks if the specified label column exists in the data; if not, it raises a "ValueError" with an appropriate error message.

The program then filters out non-numeric columns to include only numerical features in the analysis.
![Code Screenshot](screenshots/load_airbnbpng)

**Data Analysis:**
Key statistical measures, including mean, median, standard deviation, and quartiles, are calculated to provide a summary of the data distribution.

Correlation matrices and visualizations are employed to unravel relationships between variables. This helps in understanding how different features interact with each other.

Insights are extracted by identifying trends, patterns, and potential factors influencing the target variable (e.g., nightly prices). This understanding is crucial for formulating hypotheses for machine learning modeling.

#### Regression Modelling
A linear regression model using SGDRegressor from scikit-learn is trained to predict the "Price_Night" feature.

A custom grid search is implemented to find optimal hyperparameters for the regression model. This involves tuning parameters like learning rate and regularization.

Key performance metrics, including RMSE and R^2, are computed for both training and test sets to assess the baseline model's performance.

Decision trees, random forests, and gradient boosting regression models are trained and tuned for improved performance.

The best models, along with their hyperparameters and performance metrics, are saved in dedicated folders.

#### Classification Modelling

The dataset is preprocessed to handle categorical features and encode the target variable for classification.

Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier are chosen as classification models.

Hyperparameters for each classification model are tuned using the custom grid search.

Accuracy, classification report, and confusion matrix are computed to evaluate the performance of each classification model.

The best classification models, hyperparameters, and performance metrics are saved in dedicated folders.


**Visualization:** Visualizations play a crucial role in understanding the distribution, relationships, and patterns within the Airbnb dataset. Leveraging the Matplotlib library, a variety of visualizations are created to enhance data interpretation.

The visualize_predictions method generates a scatter plot showing predicted values in red and true values in blue. It includes the Root Mean Squared Error (RMSE) and R-squared values.
Confusion Matrix:

The plot_confusion_matrix method generates a confusion matrix for classification models.
ROC Curve:

The visualize_roc_curve method creates a Receiver Operating Characteristic (ROC) curve for binary classification models.
Precision-Recall Curve:

The visualize_precision_recall_curve method generates a precision-recall curve for binary classification models.


**Refactoring:** The project underwent a refactoring process to enhance code structure, improve modularity, and introduce better organization. The refactoring aimed to address several aspects:

1. **Modularization of Code:**
   - Code was organized into modular functions and classes, providing a clearer structure.
   - Distinct functionalities, such as data loading, model training, and hyperparameter tuning, were encapsulated into separate functions and classes.

2. **Code Reusability:**
   - Functions were designed to be more generic, promoting reusability across different models and datasets.
   - The `Modelling` class now serves as a foundation for both regression and classification tasks, enabling shared functionalities.

3. **Consistent Naming Conventions:**
   - Ensured consistent naming conventions for variables, functions, and classes to improve code readability.

4. **Hyperparameter Tuning:**
   - Implemented two hyperparameter tuning functions: `custom_tune_regression_model_hyperparameters` for manual tuning and `tune_regression_model_hyperparameters` utilizing SKLearn's `GridSearchCV` for automated tuning.

5. **Folder Structure:**
   - Created a `models` folder to store trained models, hyperparameters, and metrics.
   - Within the `models` folder, separate subfolders were designated for each type of model (e.g., `regression`).

6. **Documentation:**
   - Improved code comments and added explanatory comments to aid understanding.
   - Updated the README file to reflect the new structure, functions, and their purposes.

## How to Use

1. **Load Data:**
   - Ensure your data loading function or script is compatible with the structure in `modelling.py`.

2. **Train and Evaluate Models:**
   - Utilize the `evaluate_all_models` function by providing your training and validation datasets.
   - This function will train multiple regression models, tune hyperparameters, and save models along with their metrics.


![Code Screenshot](screenshots/how-to-use: train and evaluate.png)

4. **Find the Best Model:**
   - After running the `evaluate_all_models` function, use `find_best_model` to identify the best-performing model.
  
![Code Screenshot](screenshots/how-to-use: find the best model.png)

5. **Custom Hyperparameter Tuning:**
   - If desired, you can use `custom_tune_regression_model_hyperparameters` for manual hyperparameter tuning.

![Code Screenshot](screenshots/how-to-use: custom hyperparameter tuning.png)

6. **SKLearn Grid Search:**
   - Alternatively, use `tune_regression_model_hyperparameters` for automated hyperparameter tuning using SKLearn's `GridSearchCV`.
  
![Code Screenshot](screenshots/how-to-use: cGridSearch.png)

Refer to the code comments and function docstrings for more detailed information.




# Learnings
The Airbnb Data Cleaning and Analysis Project provided a valuable learning experience as it marked several first-time achievements in my data science journey. 

**Pandas:** Working on this project introduced me to the powerful Pandas library in Python. I learned to efficiently manipulate and analyze data using Pandas data structures like DataFrames and Series. The library's functions and methods streamlined data cleaning tasks and enabled smooth data exploration.

**Data Cleaning:** Prior to this project, my data cleaning exercises were limited to tutorials and practice datasets. However, working with the real-world Airbnb dataset presented practical challenges like missing values, data inconsistencies, and diverse data types. I gained hands-on experience in addressing these real data issues, enhancing my data wrangling skills significantly.

The project's focus on data cleaning, data analysis, and building a predictive model with neural networks provided a comprehensive understanding of the data science workflow. It allowed me to apply theoretical knowledge acquired through courses to tackle real-world data complexities.

Throughout the project, I learned the importance of data preprocessing and the impact it has on the quality of analysis and model performance. It provided valuable insights into data-driven decision-making and reinforced the significance of clean, reliable data for accurate predictions.

The project's hands-on nature and the exposure to a real-world dataset offered practical perspectives on data science tasks, laying a strong foundation for future data-driven projects.

## Milestone 1-3
Throughout Milestone 1, 2, and 3, I developed a data science project focused on cleaning the Airbnb data. The project encompasses multiple key functionalities and features.

I have implemented a robust data cleaning process to handle missing values, data type conversion, and inconsistencies in the raw Airbnb dataset. The cleaned dataset ensures the reliability and quality of subsequent analyses. The project allows users to specify a target label column during analysis. This customization option enables users to focus on specific aspects of the rental market based on their interests. The cleaned data is saved as "clean_tabular_data.csv" to facilitate future analysis and usage. This feature ensures the availability of a reliable and cleaned dataset for further tasks. The project is designed with user-friendliness in mind, providing easy execution of data cleaning and analysis scripts. It accommodates users with varying programming experience. To promote code reusability and maintainability, the implementation follows a modular code structure. Each major feature is encapsulated in separate functions, making the code easy to understand and modify.

**Python:** 

I have utilized Python as the primary programming language for this project. Python's versatility, extensive libraries, and strong community support make it an ideal choice for data science tasks. Python's simplicity and readability allow for rapid development and easy debugging. Additionally, its extensive ecosystem of libraries makes it an ideal choice for data science and machine learning tasks.

**Pandas:** 

Pandas is a powerful library in Python for data manipulation and analysis. I have used Pandas to handle data cleaning tasks, data exploration, and data wrangling. Pandas provides intuitive data structures (DataFrames and Series) and powerful tools for data manipulation and future analysis. It simplifies complex data cleaning tasks and enables efficient data exploration.
    
**GitHub:** 

I have utilized GitHub as the version control platform to manage the project's codebase and track changes throughout development. GitHub provides version control and collaborative features, enabling seamless collaboration among team members. Although this aspect hasnt bee used in this solo project it is good to learn these for future projects. It allows me to track changes, manage issues, and maintain a well-organized codebase.
   
**AST:** 

The ast (Abstract Syntax Trees) module in Python has been used to parse Python source code and extract information about the code's abstract syntax structure. The ast module is used to parse complex strings in the "Description" column during data cleaning. By utilizing the abstract syntax trees, we can handle intricate structures and perform data transformations effectively.

## Milestone 4:** In the process of building a regression model to predict nightly costs for Airbnb listings, the modelling.py script follows a systematic approach. Starting with the import of necessary libraries and loading data via the load_airbnb function, it utilizes the SGDRegressor class from sklearn to train a linear regression model. Subsequently, the model's performance is evaluated using metrics like RMSE and R^2 on both training and test sets, establishing a baseline for comparison. 

![Code Screenshot](screenshots/train_model.png)
![Code Screenshot](screenshots/predict_and_evaluate.png)

Hyperparameter tuning is approached with tune_regression_model_hyperparameters, and an SKLearn-based method employing GridSearchCV. 

![Code Screenshot](screenshots/tune_regression_model_hyperparameters.png)

The save_model function is designed to store the trained model, hyperparameters, and performance metrics in a structured folder system within the 'models' directory. 

![Code Screenshot](screenshots/save_model.png)

Extending beyond a simple linear model, the script explores decision trees, random forests, and gradient boosting, employing the same evaluation and tuning processes for each. The evaluate_all_models function orchestrates this process, saving each model in its dedicated folder. 

![Code Screenshot](screenshots/evaluate_all_models.png)

Finally, the find_best_model function assesses and returns the best-performing model, allowing for a comprehensive evaluation of the regression models' effectiveness. These steps provide a thorough and adaptable framework for regression model development, tuning, evaluation, and storage.

![Code Screenshot](screenshots/find_best_model.png)

## Milestone 5:** In implementing the classification tasks for the Airbnb dataset, I first loaded the data using the load_airbnb function, specifying "Category" as the label. To assess the model's performance, I computed key metrics such as the F1 score, precision, recall, and accuracy for both the training and test sets. To fine-tune the model's hyperparameters, I created the tune_classification_model_hyperparameters function, which, similar to its regression counterpart, conducts a grid search over a range of hyperparameter values, using accuracy as the evaluation metric. 

![Code Screenshot](screenshots/tune_classification_model_hyperparameters.png)

The best model, its hyperparameters, and performance metrics are then saved in a folder named after the model class within the classification directory in the models folder, specifically 'models/classification/logistic_regression'. To further enhance classification performance, I extended the evaluation to decision trees, random forests, and gradient boosting, all provided by scikit-learn. The tune_classification_model_hyperparameters function was applied to each of these models, and the results, including the models, hyperparameters, and metrics, were saved in folders corresponding to the model class within the classification directory. 

The evaluate_all_models function was adapted to accept a task_folder argument to specify the relevant directory for the classification models. 

Classification: evaluate_all_models.png
![Code Screenshot](screenshots/Classification: evaluate_all_models.png)

Finally, the find_best_model function was modified to consider the task_folder parameter, enabling it to locate and compare models within the specified classification directory. 

*LINK SCREENSHOT

This entire classification pipeline ensures that the best-performing classification model is identified, along with its hyperparameters and performance metrics. The implementation demonstrates a systematic approach to building, tuning, and evaluating classification models for the given Airbnb dataset.

**Matplotlib:**

## Milestone 6:** In this step, we set up the foundational components for training a PyTorch neural network to predict the nightly price of Airbnb listings based on tabular data. We began by creating a PyTorch Dataset named AirbnbNightlyPriceRegressionDataset, which loads the dataset from a CSV file and returns tuples of features and labels. These features represent the numerical tabular data of the Airbnb listings, while the labels correspond to the price per night. Additionally, we defined data loaders to handle the training and testing datasets, with the training set further split into training and validation sets. The neural network architecture, encapsulated within the TabularModel class, was designed to initially ingest only numerical tabular data and consists of fully connected layers with ReLU activation functions. Subsequently, we implemented the training process in the train function, performing forward passes, loss calculation, and optimization of model parameters using gradient descent. Hyperparameters, such as the optimizer, learning rate, and network architecture, were specified in a YAML configuration file. Lastly, we introduced functions for evaluating model performance, saving models and their associated metadata, and conducting hyperparameter tuning to find the optimal network configuration. This step laid the groundwork for subsequent iterations and improvements to the neural network model.

**Neural Network:**

The neural network model is defined in the TabularModel class. Currently, it only ingests the numerical tabular data. The architecture consists of fully connected layers (linear layers) with ReLU activation functions.

![Code Screenshot](screenshots/TabularModel_Class.png)

The training process is defined in the train function. It takes the model, data loader, number of epochs, learning rate, and an optional configuration file (nn_config.yaml). The function performs a forward pass on a batch of data, calculates the loss, and optimizes the model parameters using gradient descent. The training loop iterates through every batch in the dataset for the specified number of epochs and evaluates the model performance on the validation dataset after each epoch. TensorBoard is used to visualize the training curves and accuracy on both the training and validation sets.

![Code Screenshot](screenshots/train.png)

The neural network architecture and hyperparameters are specified in a YAML file called nn_config.yaml. It defines:

- The name of the optimizer used (optimiser)
- The learning rate
- The width of each hidden layer (hidden_layer_width)
- The depth of the model

![Code Screenshot](screenshots/nn_config.png)

The save_model function saves the trained model, hyperparameters, and performance metrics to files. Performance metrics include RMSE loss, R^2 score, training duration, and inference latency.

To find the best neural network configuration, the find_best_nn function trains models with different configurations specified in nn_config.yaml. It sequentially trains models with each configuration, saves the configuration used, and returns the best model, metrics, and hyperparameters.



**Further Development:**
**Version 2.0:**
