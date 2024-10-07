# Ophthalmologic Data Analysis

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Goal](#goal)
- [Preprocessing](#preprocessing)
- [Machine Learning Models](#machine-learning-models)
  - [Logistic Regression with ADASYN](#logistic-regression-with-adasyn)
  - [Random Forest with ADASYN](#random-forest-with-adasyn)
  - [SVM with ADASYN](#svm-with-adasyn)
  - [SVM with SMOTE](#svm-with-smote)
  - [SVM with RandomOverSampler](#svm-with-randomoversampler)
  - [SVM with SMOTE-ENN](#svm-with-smote-enn)
- [Results](#results)
- [Usage](#usage)
- [Contact](#contact)

## Introduction

This project focuses on analyzing ophthalmologic data using various machine learning techniques. The primary goal is to classify patients into different classes based on their ophthalmologic data. The data preprocessing steps, machine learning models, and the application of different oversampling techniques are thoroughly documented.

## Dataset Description

The dataset used in this project is an ophthalmologic dataset. It includes features extracted from patients' ophthalmologic exams, with the aim of classifying them into different categories:

- `CLASE`: The original class label.
- `OJO`: Eye identifier.
- `USER`: User identifier.
- Additional features representing various ophthalmologic metrics.

### New Classes

The original classes are redefined as follows:
- Class 1 remains Class 1.
- Classes 2, 3, and 4 are merged into a new Class 2.

## Goal

The primary goal of this project is to classify patients into two classes using various machine learning models and oversampling techniques to handle class imbalance. The models evaluated include Logistic Regression, Random Forest, and SVM. The oversampling techniques applied include ADASYN, SMOTE, RandomOverSampler, and SMOTE-ENN.

## Preprocessing

The preprocessing steps include:
1. Loading the dataset.
2. Creating new classes.
3. Splitting the data into training and testing sets.

## Machine Learning Models

### Logistic Regression with ADASYN

Logistic Regression is trained using the ADASYN oversampling technique to handle class imbalance. The steps include cross-validation, training, and evaluation.

### Random Forest with ADASYN

Random Forest classifier is applied with the ADASYN oversampling technique. The model undergoes cross-validation, training, and evaluation.

### SVM with ADASYN

Support Vector Machine (SVM) is trained using the ADASYN oversampling technique. Cross-validation is performed to find the best parameters, followed by training and evaluation.

### SVM with SMOTE

SVM is applied with the SMOTE oversampling technique. Cross-validation, training, and evaluation are performed.

### SVM with RandomOverSampler

SVM is used with RandomOverSampler to handle class imbalance. The model undergoes cross-validation, training, and evaluation.

### SVM with SMOTE-ENN

SVM is trained using the SMOTE-ENN technique, which combines oversampling and undersampling to handle class imbalance. Cross-validation, training, and evaluation are performed.

## Results

The results of the different models and oversampling techniques are combined and sorted by macro accuracy. The final results are saved in a CSV file.
