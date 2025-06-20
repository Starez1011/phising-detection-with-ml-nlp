# Phishing Website Detection

A machine learning-based system for detecting phishing websites using various features extracted from URLs and website characteristics.

## Project Overview

This project implements a machine learning system to detect phishing websites by analyzing various features of URLs and website characteristics. The system uses multiple machine learning models to classify websites as either legitimate or phishing.

## Features

The system analyzes the following features to detect phishing websites:

1. **URL-based Features**:

   - Long URL detection (URLs longer than 54 characters)
   - Presence of @ symbol in URL
   - Presence of redirection (//) in URL
   - Prefix/suffix separation in domain
   - Number of sub-domains
   - Presence of IP address in URL
   - Use of URL shortening services
   - Presence of HTTPS token

2. **Domain-based Features**:

   - Web traffic analysis
   - Domain registration length
   - DNS record presence
   - Age of domain

3. **Statistical Features**:
   - Website traffic ranking
   - Domain age verification
   - DNS record validation

## Model Training and Performance

The project implements and compares multiple machine learning models:

1. **Naive Bayes**

   - Training Accuracy: 67.2%
   - Test Accuracy: 68.4%
   - Training Time: 0.004s
   - Good for quick initial screening
   - Confusion Matrix:
     ```
     [[1152   25]
      [ 696  408]]
     ```
   - Working Principle:
     - Uses Bayes' theorem to calculate probability of a website being phishing
     - Assumes features are independent of each other
     - Fast training but may oversimplify complex relationships

2. **Decision Tree**

   - Training Accuracy: 78.5%
   - Test Accuracy: 77.1%
   - Training Time: 0.007s
   - Balanced performance and speed
   - Confusion Matrix:
     ```
     [[920 257]
      [265 839]]
     ```
   - Working Principle:
     - Creates a tree-like model of decisions
     - Each node represents a feature test
     - Leaf nodes represent class labels
     - Easy to interpret but can overfit

3. **Random Forest**

   - Training Accuracy: 78.5%
   - Test Accuracy: 77.2%
   - Training Time: 0.282s
   - Best overall performance
   - Confusion Matrix:
     ```
     [[918 259]
      [262 842]]
     ```
   - Working Principle:
     - Ensemble of multiple decision trees
     - Each tree trained on random subset of data
     - Final prediction by majority voting
     - Reduces overfitting through averaging
     - Handles both numerical and categorical features well

4. **XGBoost**
   - Training Accuracy: 78.4%
   - Test Accuracy: 77.0%
   - Training Time: 1.260s
   - Good performance but slower training
   - Confusion Matrix:
     ```
     [[914 263]
      [261 843]]
     ```
   - Working Principle:
     - Gradient boosting framework
     - Sequentially builds weak learners
     - Each new model corrects errors of previous ones
     - Uses advanced regularization to prevent overfitting
     - Handles missing values automatically

### Model Selection and Multi-Model Approach

The project uses a multi-model approach with Random Forest and XGBoost as the primary models because:

1. **Complementary Strengths**:

   - Random Forest: Better at handling categorical features and outliers
   - XGBoost: Better at capturing complex patterns and relationships

2. **Ensemble Benefits**:

   - Reduces variance in predictions
   - Improves overall accuracy
   - More robust to different types of phishing attempts

3. **Why This Combination is Better**:

   - Random Forest provides good baseline performance
   - XGBoost captures patterns that Random Forest might miss
   - Together they provide more reliable predictions
   - Reduces false positives and false negatives
   - Better handling of edge cases

4. **Performance Comparison**:

   - Random Forest:

     - Precision: 0.78 (legitimate), 0.76 (phishing)
     - Recall: 0.78 (legitimate), 0.76 (phishing)
     - F1-Score: 0.78 (legitimate), 0.76 (phishing)

   - XGBoost:
     - Precision: 0.78 (legitimate), 0.76 (phishing)
     - Recall: 0.78 (legitimate), 0.76 (phishing)
     - F1-Score: 0.78 (legitimate), 0.76 (phishing)

5. **Confusion Matrix Analysis**:

   - Random Forest:

     - True Negatives (Legitimate): 918
     - False Positives: 259
     - False Negatives: 262
     - True Positives (Phishing): 842

   - XGBoost:
     - True Negatives (Legitimate): 914
     - False Positives: 263
     - False Negatives: 261
     - True Positives (Phishing): 843

   This shows that both models have similar performance but make different types of errors, making them complementary when used together.

## Usage

1. **GUI Application**:
   - Launch the GUI application
   - Enter the URL to check
   - Click "Analyze" to get results

## Acknowledgments

- Dataset sources - Kaggle, Phistank
