# Heart-Disease-Prediction 
This project predicts heart disease using Logistic Regression on the  [Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

Features  
- Load and explore dataset (shape, class distribution, mean values per class)  
- Standardize feature values using StandardScaler  
- Split dataset into train and test sets (80/20) with stratification  
- Train a Logistic Regression classifier  
- Evaluate the model using:  
  - Accuracy  
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-score)  
- Make predictions for new inputs via a reusable function  
- Save the trained model and scaler using pickle  

Results  
- Training Accuracy: ~0.86  
- Test Accuracy: ~0.80  
- Confusion matrix and classification report show good recall for heart disease detection, with minor false positives/negatives.  
