import pandas as pd
import joblib

# Bước 1: Tải mô hình đã lưu
model = joblib.load('logistic_regression_model.pkl')

# Bước 2: Nhập dữ liệu đầu vào
input_data = {
    'temperature': [float(input("Nhập nhiệt độ: "))],
    'vibration': [float(input("Nhập độ rung: "))],
    'pressure': [float(input("Nhập áp suất: "))],
    'operating_time': [float(input("Nhập thời gian hoạt động: "))],
    'maintenance_history': [int(input("Nhập lịch sử bảo trì (0 hoặc 1): "))]
}

# Chuyển đổi dữ liệu đầu vào thành DataFrame
input_df = pd.DataFrame(input_data)

# Bước 3: Dự đoán
prediction = model.predict(input_df)

# Bước 4: Hiển thị kết quả
if prediction[0] == 1:
    print("Dự đoán: Cần bảo trì")
else:
    print("Dự đoán: Không cần bảo trì")