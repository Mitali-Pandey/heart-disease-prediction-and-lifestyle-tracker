"""
Data Cleaning and Preprocessing Module
Handles missing values, outliers, duplicates, and data quality checks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Comprehensive data cleaning and preprocessing class"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_report = {}
        
    def detect_duplicates(self):
        """Detect and report duplicate rows"""
        duplicates = self.df.duplicated().sum()
        self.cleaning_report['duplicates'] = {
            'count': duplicates,
            'percentage': (duplicates / len(self.df)) * 100
        }
        return duplicates
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        self.cleaning_report['duplicates_removed'] = before - after
        return self.df
    
    def detect_missing_values(self):
        """Detect and report missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        self.cleaning_report['missing_values'] = missing_df.to_dict('records')
        return missing_df
    
    def handle_missing_values(self, strategy='mean'):
        """Handle missing values based on strategy"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            for col in numeric_cols:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            self.df = self.df.dropna()
        
        self.cleaning_report['missing_strategy'] = strategy
        return self.df
    
    def detect_outliers_zscore(self, threshold=3):
        """Detect outliers using Z-score method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col]))
            outliers = np.where(z_scores > threshold)[0]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'indices': outliers.tolist()
            }
        
        self.cleaning_report['outliers_zscore'] = outlier_info
        return outlier_info
    
    def detect_outliers_iqr(self):
        """Detect outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        self.cleaning_report['outliers_iqr'] = outlier_info
        return outlier_info
    
    def remove_outliers_zscore(self, threshold=3):
        """Remove outliers using Z-score method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        before = len(self.df)
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col]))
            self.df = self.df[z_scores < threshold]
        
        after = len(self.df)
        self.cleaning_report['outliers_removed'] = before - after
        return self.df
    
    def get_correlation_matrix(self):
        """Generate correlation matrix"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        return correlation_matrix
    
    def plot_correlation_heatmap(self, figsize=(12, 10)):
        """Plot correlation heatmap"""
        correlation_matrix = self.get_correlation_matrix()
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return plt
    
    def get_cleaning_summary(self):
        """Get comprehensive cleaning summary"""
        summary = {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'cleaning_report': self.cleaning_report
        }
        return summary
    
    def get_cleaned_data(self):
        """Return cleaned dataframe"""
        return self.df


def load_sample_data():
    """
    Load real-world UCI/Kaggle heart dataset if present, else generate synthetic data.

    Expected UI schema:
      age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

    If you drop a Kaggle CSV into `heart disease/data/`, this function will try to:
    - map/rename common column names
    - coerce feature columns to numeric
    - convert UCI-style encodings (cp 1-4 -> 0-3, slope 1-3 -> 0-2, thal {3,6,7} -> 0-2)
    - make `target` binary (0/1)
    """

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    preferred_filenames = [
        "heart_combined.csv",
        "heart_uci.csv",
        "heart.csv",
        "heart_disease_dataset.csv",
        "heart-disease-dataset.csv",
        "heart_disease.csv",
    ]

    # Merge all CSVs found in this folder (deterministic order).
    csv_paths = []
    if os.path.isdir(data_dir):
        csv_paths = [
            os.path.join(data_dir, fname)
            for fname in sorted(os.listdir(data_dir))
            if fname.lower().endswith(".csv")
        ]

    def _make_synthetic():
        np.random.seed(42)
        n_samples = 1000
        data = {
            "age": np.random.randint(29, 80, n_samples),
            "sex": np.random.randint(0, 2, n_samples),
            "cp": np.random.randint(0, 4, n_samples),
            "trestbps": np.random.randint(94, 200, n_samples),
            "chol": np.random.randint(126, 564, n_samples),
            "fbs": np.random.randint(0, 2, n_samples),
            "restecg": np.random.randint(0, 3, n_samples),
            "thalach": np.random.randint(71, 202, n_samples),
            "exang": np.random.randint(0, 2, n_samples),
            "oldpeak": np.random.uniform(0, 6.2, n_samples),
            "slope": np.random.randint(0, 3, n_samples),
            "ca": np.random.randint(0, 4, n_samples),
            "thal": np.random.randint(0, 3, n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }

        df_syn = pd.DataFrame(data)
        n_age = (df_syn["age"] > 60).sum()
        n_chol = (df_syn["chol"] > 250).sum()
        n_bps = (df_syn["trestbps"] > 140).sum()
        df_syn.loc[df_syn["age"] > 60, "target"] = np.random.choice(
            [0, 1], p=[0.3, 0.7], size=n_age
        ).astype(np.int32)
        df_syn.loc[df_syn["chol"] > 250, "target"] = np.random.choice(
            [0, 1], p=[0.4, 0.6], size=n_chol
        ).astype(np.int32)
        df_syn.loc[df_syn["trestbps"] > 140, "target"] = np.random.choice(
            [0, 1], p=[0.4, 0.6], size=n_bps
        ).astype(np.int32)
        return df_syn

    if not csv_paths:
        # No real dataset present yet.
        return _make_synthetic()

    print(f"Merging {len(csv_paths)} real CSV file(s) for training...")
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    # Rename common variants to match UI schema.
    rename_map = {
        "num": "target",
        "heart_disease": "target",
        "HeartDisease": "target",
        "target_label": "target",
        # Feature name variants seen in common Kaggle heart datasets
        "ChestPainType": "cp",
        "RestingBP": "trestbps",
        "Resting Blood Pressure": "trestbps",
        "Cholesterol": "chol",
        "Total Cholesterol": "chol",
        "FastingBS": "fbs",
        "Fasting Blood Sugar": "fbs",
        "RestingECG": "restecg",
        "Resting ECG": "restecg",
        "MaxHR": "thalach",
        "Max Heart Rate": "thalach",
        "ExerciseAngina": "exang",
        "Exercise Angina": "exang",
        "Oldpeak": "oldpeak",
        "ST_Slope": "slope",
        "ST slope": "slope",
        "MajorVessels": "ca",
        "Major Vessels": "ca",
        "Thalassemia": "thal",
        "Thal": "thal",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # If target isn't binary, binarize it.
    if "target" not in df.columns:
        raise ValueError(
            "CSV loaded but it doesn't contain a 'target' column (or known equivalent like 'num')."
        )

    # Coerce feature columns to numeric; drop rows that can't be converted.
    feature_cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Force numeric conversion for all required columns.
    # First, handle string categories for key categorical inputs.
    # cp (chest pain type)
    if df["cp"].dtype == object:
        cp_map = {
            "Typical Angina": 0,
            "Typical angina": 0,
            "Atypical Angina": 1,
            "Atypical angina": 1,
            "Non-anginal Pain": 2,
            "Non-anginal": 2,
            "Non-anginal pain": 2,
            "Asymptomatic": 3,
            "asymptomatic": 3,
        }
        df["cp"] = df["cp"].map(cp_map)

    # restecg (resting ECG)
    if df["restecg"].dtype == object:
        restecg_map = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "ST-T wave abnormality": 1,
            "Left Ventricular Hypertrophy": 2,
            "Left ventricular hypertrophy": 2,
        }
        df["restecg"] = df["restecg"].map(restecg_map)

    # exang (exercise induced angina)
    if df["exang"].dtype == object:
        exang_map = {
            "Yes": 1,
            "No": 0,
            "TRUE": 1,
            "FALSE": 0,
            "True": 1,
            "False": 0,
        }
        df["exang"] = df["exang"].map(exang_map)

    # fbs (fasting blood sugar > 120)
    if df["fbs"].dtype == object:
        fbs_map = {
            "Yes": 1,
            "No": 0,
            "TRUE": 1,
            "FALSE": 0,
            "True": 1,
            "False": 0,
        }
        df["fbs"] = df["fbs"].map(fbs_map)

    # slope (ST segment slope)
    if df["slope"].dtype == object:
        slope_map = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2,
        }
        df["slope"] = df["slope"].map(slope_map)

    # thal (thalassemia / perfusion category)
    if df["thal"].dtype == object:
        thal_map = {
            "Normal": 0,
            "Fixed Defect": 1,
            "Reversible Defect": 2,
            "fixed defect": 1,
            "reversible defect": 2,
        }
        df["thal"] = df["thal"].map(thal_map)

    # Now coerce everything to numeric.
    for col in feature_cols + ["target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_cols + ["target"]).copy()

    # Convert UCI encodings (common in Kaggle/UCI heart datasets) -> UI encodings (0-based).
    # cp: typically 1..4 in UCI; UI expects 0..3.
    if df["cp"].min() >= 1 and df["cp"].max() <= 4:
        df["cp"] = df["cp"] - 1

    # slope: typically 1..3 in UCI; UI expects 0..2.
    if df["slope"].min() >= 1 and df["slope"].max() <= 3:
        df["slope"] = df["slope"] - 1

    # thal: typically {3,6,7}; UI expects 0..2.
    if set(df["thal"].dropna().unique()).issubset({3, 6, 7}):
        df["thal"] = df["thal"].map({3: 0, 6: 1, 7: 2})
    elif df["thal"].min() >= 1 and df["thal"].max() <= 3:
        # Some Kaggle versions normalize thal to 1..3 already.
        df["thal"] = df["thal"] - 1

    # restecg sometimes comes in as 1..3 instead of 0..2
    if df["restecg"].min() >= 1 and df["restecg"].max() <= 3:
        df["restecg"] = df["restecg"] - 1

    # ca: sometimes 0..4 (where 4 indicates missing); UI expects 0..3.
    if df["ca"].max() > 3:
        # Clip higher value(s) to 3 to keep UI range stable.
        df["ca"] = df["ca"].clip(upper=3)

    # target: sometimes 0..4 (num) in UCI; make it binary 0/1.
    uniq = sorted(df["target"].unique())
    if uniq and (min(uniq) >= 0 and max(uniq) > 1):
        df["target"] = (df["target"] > 0).astype(int)
    else:
        # ensure 0/1
        df["target"] = df["target"].astype(int)

    print(f"Real dataset rows after cleaning: {len(df)}")
    return df

