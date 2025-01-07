# Multi-Input Neural Network for COVID-19 Detection from Audio Signals

## Project Description
The COVID-19 Detection System employs a multi-branch convolutional neural network (CNN) to analyze audio recordings for signs of COVID-19. By extracting relevant features from audio samples and addressing class imbalance through SMOTEENN, the model aims to achieve high accuracy in detecting COVID-19 cases based on audio characteristics.

## Dataset
The dataset consists of audio recordings categorized by COVID-19 status. Each audio type (breathing, cough, speech) is organized into distinct directories. The recordings are standardized to ensure uniform duration, with specific sampling rates applied for each type:
Breathing samples: 80,000 Hz
Cough recordings: 96,000 Hz
Speech recordings: 160,000 Hz

## Methodology
###  Data Preprocessing
Audio Standardization: Audio files are organized and standardized to ensure consistent duration across different types.
Class Imbalance Handling: The SMOTEENN technique is applied to balance the dataset, addressing both minority class underrepresentation and noise reduction.
###   Feature Engineering
The feature extraction process captures both spectral and temporal characteristics:
- Mel Spectrograms: Generated using the librosa library, these features undergo logarithmic power transformation and normalization.
- Zero-Crossing Rate: Captures variations in signal polarity.
- Spectral Centroid: Quantifies the brightness of the audio signals.

###   Model Architecture
The model utilizes a multi-input CNN with three parallel branches for processing spectral features and a dense network for auxiliary features:
CNN Branches: Each branch processes mel spectrograms through convolutional subnetworks with batch normalization and dropout regularization.
Feature Fusion: Outputs from all branches are concatenated for final classification through a sigmoid activation layer.
###   Training
The model is trained on a balanced dataset with a batch size of 32 over 20 epochs. Early stopping is implemented to prevent overfitting.
###   Evaluation Metrics
Key metrics include accuracy, precision, recall, F1-score, and confusion matrices to assess model performance.

## Results
The model achieved the following performance metrics:
| Metric            | Train  | Test   |
|-------------------|--------|--------|
| Balanced Accuracy | 65.12% | 63.50% |
| F-1 Score         | 76.95% | 37.29% |
| Precision         | 73.98% | 24.14% |
| Recall            | 80.24% | 82.35% |

While the training dataset performed well in identifying true positives, generalization to the test dataset revealed challenges with false positives.
## Future Work
- Dataset Expansion: Acquire more diverse audio samples to improve model generalization.
- Feature Augmentation: Experiment with additional audio features or pre-trained models.
- Model Refinement: Fine-tune hyperparameters and explore alternative architectures for better performance.

## Contributors
- Shazia Muckram
- Ophelia Sin 
