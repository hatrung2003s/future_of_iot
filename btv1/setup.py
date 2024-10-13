# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # Thư viện để lưu và tải mô hình

# Bước 1: Đọc dữ liệu từ file CSV
data = pd.read_csv('/home/admin/Desktop/BAI16/data1.csv')

# Bước 2: Tiền xử lý dữ liệu
data['Light Status'] = data['Light Status'].map({'On': 1, 'Off': 0})
data['Light Sensor Status'] = data['Light Sensor Status'].map({'On': 1, 'Off': 0})

data['Time'] = pd.to_datetime(data['Time'], format='%H:%M', errors='coerce').dt.time
data['Hour'] = data['Time'].apply(lambda x: x.hour)
data['Minute'] = data['Time'].apply(lambda x: x.minute)

day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}
data['DayOfWeek'] = data['Day of the Week'].map(day_mapping)

data = data.drop(columns=['Day of the Week', 'Time'])

X = data.drop('Light Status', axis=1)
y = data['Light Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 5: Xây dựng mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Bước 6: Dự đoán và tính độ chính xác
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác của mô hình: {accuracy * 100:.2f}%")

# Lưu mô hình vào file
joblib.dump(model, 'light_model.pkl')
print("Mô hình đã được lưu thành công.")
