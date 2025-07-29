# Airbnb New User Booking Prediction

Predict where a new Airbnb user will book their first trip based on demographic and session data. This end-to-end machine learning pipeline covers data cleaning, feature engineering, class imbalance handling, model building with PyTorch, and deployment via FastAPI.

---

## Problem Statement

Airbnb wants to personalize user experiences by predicting which country a new user is likely to book in. The goal is to reduce time-to-booking and improve engagement by anticipating user destinations.

There are 12 possible country labels, including:
- Destination countries like `US`, `FR`, `IT`, `GB`, etc.
- `NDF`: No destination found (user didn’t book)
- `other`: Booked, but country not in known list

---

## Datasets Used

- **train_users.csv / test_users.csv**: User demographics, account creation, booking labels
- **sessions.csv**: User interaction logs (actions, devices, time spent)

---

## EDA & Data Cleaning

### Sessions Dataset
- **Imputed missing action columns** based on correlated values (e.g., `action_type`).
- **Handled missing `secs_elapsed`** via type-specific means; removed outliers using Z-score.
- Created features like:
  - `total_session_duration`
  - `most_used_device`
  - `total_actions`
  - Normalized frequencies of selected action categories

### Train Users Dataset
- Cleaned `age` outliers and birth-year entries
- Preserved unknown gender values for signal
- Mapped rare categorical values to `"other"`
- Parsed date features and extracted meaningful components (day, month, weekday, etc.)

---

## Feature Engineering

- Combined user and session data on `user_id`
- Applied **target encoding** with rare grouping to reduce dimensionality
- Saved mappings and encoders for inference

---

## Modeling

- **Architecture**: Simple feedforward neural network in PyTorch  
  - 2 hidden layers (256 units)  
  - BatchNorm + ReLU activations  
- **Loss Function**: Focal Loss  
  - Tuned `alpha=0.25`, `gamma=2` to address heavy class imbalance
- **Sampling Strategy**:  
  - Tried undersampling, hybrid → reduced performance  
  - Final choice: **Random oversampling** via `imbalanced-learn`

---

## Results

| Metric     | Value   |
|------------|---------|
| Precision  | 0.557   |
| Recall     | 0.679   |
| F1-Score   | 0.575   |

---

## Deployment

- Deployed using **FastAPI** with HTML frontend
- Modular, object-oriented pipeline for preprocessing and inference
- Dockerized the entire project for consistent deployment  
  *(Note: Large image size prevented Docker Hub push)*

### ▶Run Locally
```bash
# Start FastAPI server
python -m uvicorn src.main:app --reload
