import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

print("Đang bắt đầu quá trình huấn luyện mô hình...")

# 1. Load dataset
try:
    df = pd.read_csv('train_filtered.csv')
    print(f"Dataset đã được load: {df.shape[0]} dòng, {df.shape[1]} cột.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'train_filtered.csv'. Vui lòng kiểm tra lại.")
    exit(1)

# 2. Định nghĩa đặc trưng (X) và mục tiêu (y)
features = [
    'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
    'Workout_Frequency (days/week)', 'Experience_Level'
]
targets = ['Calories', 'Proteins', 'Fats', 'Carbs']

X = df[features]
y = df[targets]

# 3. Phân chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Xây dựng pipeline tiền xử lý
numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Workout_Frequency (days/week)', 'Experience_Level']
categorical_features = ['Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# --- Mô hình 1: Hồi quy Tuyến tính (Để so sánh) ---
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', MultiOutputRegressor(LinearRegression()))])
print("Đang huấn luyện Linear Regression...")
lr_model = lr_pipeline.fit(X_train, y_train)
print("Hoàn tất.")

# --- Mô hình 2: Rừng Ngẫu nhiên (Mô hình chính) ---
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
print("Đang huấn luyện Random Forest...")
rf_model = rf_pipeline.fit(X_train, y_train)
print("Hoàn tất.")

# --- Đánh giá mô hình ---
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
print(f"[Linear Regression]  R2 Score: {r2_lr:.4f} | MAE: {mae_lr:.2f}")
print(f"[Random Forest]      R2 Score: {r2_rf:.4f} | MAE: {mae_rf:.2f}")

# ===================================================================
# LƯU MODEL ĐỂ DEPLOY
# ===================================================================
try:
    # Lưu mô hình Random Forest vì có R2 Score cao hơn
    with open('nutrition_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"\nĐÃ LƯU MÔ HÌNH (Random Forest) VÀO FILE: nutrition_model.pkl")
except Exception as e:
    print(f"\nLỖI KHI LƯU MODEL: {e}")