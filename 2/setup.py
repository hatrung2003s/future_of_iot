import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
# Bước 1: Đọc dữ liệu từ file CSV
data = pd.read_csv('2.csv')
# Kiểm tra dữ liệu
print("Dữ liệu đầu vào:")
# Bước 2: Tách biến độc lập (X) và biến phụ thuộc (y)
X = data[['temperature', 'vibration', 'pressure', 'operating_time', 'maintenance_history']]
y = data['need_maintenance']  # Cột này nên có giá trị 0 hoặc 1
# Bước 3: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Bước 4: Xây dựng mô hình Logistic Regression
model = LogisticRegression()
# Huấn luyện mô hình
model.fit(X_train, y_train)
# Bước 5: Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Bước 6: Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f'Độ chính xác: {accuracy}\n')
# Bước 8: Lưu mô hình
joblib.dump(model, 'logistic_regression_model.pkl')
print("Mô hình đã được lưu")