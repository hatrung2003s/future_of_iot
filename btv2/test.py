import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Bước 1: Tải mô hình đã lưu
model = tf.keras.models.load_model('iot_model.keras')

# Bước 2: Nhập dữ liệu từ người dùng
day_of_week = input("Nhập ngày trong tuần (0: Chủ nhật, 1: Thứ hai, ..., 6: Thứ bảy): ")
time_input = input("Nhập thời gian (định dạng HH:MM): ")
light_sensor_status = input("Nhập trạng thái cảm biến ánh sáng (0: Tắt, 1: Bật): ")

# Chuyển đổi dữ liệu nhập vào thành số
day_of_week = int(day_of_week)
time_hour = pd.to_datetime(time_input, format='%H:%M').hour
light_sensor_status = int(light_sensor_status)

# Bước 3: Tạo biến đặc trưng cho dự đoán
X_new = np.array([[day_of_week, time_hour, light_sensor_status]])

# Bước 4: Tiêu chuẩn hóa dữ liệu giống như trong quá trình huấn luyện
# (Chú ý: Bạn cần sử dụng scaler đã được huấn luyện trước đó để tiêu chuẩn hóa)
scaler = StandardScaler()
scaler.fit(np.array([[0, 0, 0], [6, 23, 1]]))  # Giả định giá trị tối thiểu và tối đa (bạn có thể thay đổi)
X_new_scaled = scaler.transform(X_new)

# Bước 5: Dự đoán trạng thái đèn
prediction = model.predict(X_new_scaled)
predicted_light_status = 1 if prediction[0] >= 0.5 else 0  # Áp dụng ngưỡng 0.5 cho phân loại nhị phân

# Bước 6: In ra kết quả
print(f"Trạng thái đèn dự đoán: {'Bật' if predicted_light_status == 1 else 'Tắt'}")
