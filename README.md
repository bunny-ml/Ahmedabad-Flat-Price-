# 🏠 Ahmedabad Flat Price Prediction

This project aims to predict the prices of flats in Ahmedabad, India, using machine learning techniques. The dataset was sourced from Kaggle and contains various features related to flats available for sale in the city.

---

## 📊 Dataset

The dataset used in this project is the [Ahmedabad Flat Price Dataset (Uncleaned)](https://www.kaggle.com/datasets/prayeshgodhani04/ahmedabad-flat-price-dataset-uncleaned) by Prayesh Godhani. It includes detailed information about flats available for sale in Ahmedabad, capturing various aspects such as:

- **Location**: Area and locality
- **Size**: Carpet area, built-up area, and super built-up area
- **Configuration**: Number of bedrooms, bathrooms, and balconies
- **Furnishing**: Type of furnishing (e.g., Semi, Full, None)
- **Floor**: Floor number and total floors
- **Price**: Listed price of the flat

*Note: The dataset is uncleaned and may require preprocessing before use.*

---

## 🧹 Data Preprocessing

The raw dataset underwent several preprocessing steps:

- **Handling missing values**: Imputed or removed missing data
- **Feature engineering**: Created new features like `area_sqft` (super built-up area)
- **Log transformation**: Applied `np.log1p()` to skewed features like `area_sqft` and `price` to reduce skewness
- **Encoding categorical variables**: Used one-hot encoding for categorical features
- **Scaling numerical features**: Standardized numerical features using `StandardScaler`

---

## 🤖 Model Development

The project employed the following machine learning techniques:

- **Model**: XGBoost Regressor
- **Hyperparameter Tuning**:
  - **RandomizedSearchCV**: Explored a wide range of hyperparameters
  - **GridSearchCV**: Fine-tuned the model with a focused search around the best parameters from RandomizedSearchCV
- **Cross-Validation**: Used 5-fold cross-validation to evaluate model performance

---

## 📈 Evaluation Metrics

The model's performance was evaluated using the following metrics:

- **R² Score**: 0.9129
- **Mean Absolute Error (MAE)**: 0.1539
- **Root Mean Squared Error (RMSE)**: 0.2949
- **Mean Absolute Percentage Error (MAPE)**: 0.0094

These metrics indicate a strong predictive performance of the model.

---

## 🌐 Streamlit Web Application

A user-friendly web application was developed using Streamlit to allow users to input flat details and receive price predictions. The application includes:

- Dropdown menus for categorical features (e.g., furnishing type, area type, transaction type)
- Numeric inputs for continuous features (e.g., number of bedrooms, size)
- Real-time price prediction upon user input

---

## 🛠️ Setup and Usage

To run this project locally:

1. Clone the repository:

   ```bash
   git clone git@github.com:bunny-ml/Ahmedabad-Flat-Price-.git
   cd Ahmedabad-Flat-Price-
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

3. Run the Streamlit application:
    
    ```bash
    streamlit run app.py


4. Open your browser and navigate to the provided local URL to interact with the application.

📁 Project Structure

    ahmedabad-flat-price-prediction/
    │
    ├── dataset/
    │   └── ahmedabad.csv                    # Raw dataset
    ├── model/                               # here is the model stored 
    |   └── categorical_features.json       
    |   └── numeric_features.json
    |   └── xgboost_pipeline_model.pkl       # model
    ├── notebooks/
    │   └── 01_data_cleaning_code.ipynb     # Data cleaning 
    │   └── 02_modeling.ipynb               # modeling / prediction
    ├── app.py                              # Streamlit application
    ├── requirements.txt                    # Project dependencies
    └── README.md                           # Project documentation
