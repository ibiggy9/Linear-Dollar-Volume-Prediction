# Product Volume Prediction

This repository contains a Linear Regression model used to predict the volume of "Cheez It" based on various other snack products.

## Description

The model ingests a dataset with sales volume of various snacks and aims to predict the volume of "Cheez It" based on this data. The model is trained multiple times to ensure that the best accuracy is achieved.

## Features

- **Data Loading and Preprocessing:** Read the dataset and prepare it for the model.
- **Linear Regression Model:** Utilizes Scikit-Learn to train a linear regression model.
- **Model Accuracy:** Assesses model accuracy using metrics like Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error.
- **Model Saving:** The best-performing model is saved for potential future use.
- **Cannibalization Scores:** Presents how much volume "Cheez It" sources from each other snack in the dataset.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/ibiggy9/Machine-Learning-Application-Canabalization-Prediction.git
    ```
2. **Install Required Libraries:** 
    ```bash
    pip install pandas numpy scikit-learn
    ```

## Usage

To run the main script (assuming it's named `cannibalization.py`):

```bash
python main.py
