# Sentiment Analysis on IMDB Movie Reviews

## Project Overview
This project focuses on performing sentiment analysis on IMDB movie reviews using deep learning techniques with an LSTM (Long Short-Term Memory) model. The dataset used contains 50,000 movie reviews labeled as positive or negative.

## Technologies Used
- Python
- TensorFlow/Keras
- Pandas
- Scikit-learn
- Kaggle API

## Dataset
The dataset used in this project is the **IMDB Dataset of 50K Movie Reviews**, which was downloaded from Kaggle.

## Steps Involved

1. **Dataset Downloading:**
   - The dataset is downloaded using the Kaggle API.
   - The zip file is extracted to access the CSV file.

2. **Data Preprocessing:**
   - The dataset is loaded using Pandas.
   - Sentiment labels are converted to numerical values (positive: 1, negative: 0).

3. **Train-Test Splitting:**
   - The dataset is split into training (80%) and testing (20%) sets.

4. **Tokenization and Padding:**
   - Tokenization is applied to convert text to sequences.
   - Padding is used to ensure uniform sequence length.

5. **LSTM Model Building:**
   - An LSTM model is created with the following layers:
     - Embedding layer
     - LSTM layer
     - Dense output layer with a sigmoid activation function
   - Model is compiled using binary cross-entropy loss and the Adam optimizer.

6. **Model Training:**
   - The model is trained with 5 epochs and a batch size of 64.
   - Validation split of 20% is used.

7. **Model Evaluation:**
   - The model is evaluated on the test data.
   - Accuracy and loss metrics are reported.

8. **Prediction Function:**
   - A function is implemented to predict sentiment based on user input reviews.

## Results
- The model achieved satisfactory accuracy on the test set.
- Example predictions:
  - "This movie was not so interesting." -> Negative
  - "This movie was very amazing." -> Positive

## How to Run the Project
1. Clone the repository from GitHub.
2. Install required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to train and evaluate the model.

## Future Improvements
- Increase the dataset size to enhance model performance.
- Tune hyperparameters for better accuracy.
- Experiment with different neural network architectures.

## License
This project is under the MIT License.

## Author
**Arpan Pramanik**

