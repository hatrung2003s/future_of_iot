import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Bước 1: Đọc dữ liệu từ CSV
data = pd.read_csv('/home/admin/Desktop/BAI14/data1.csv')  # Thay đổi tên file nếu cần
# Bước 2: Tiền xử lý dữ liệu
# Chuyển đổi trạng thái Light Status và Light Sensor Status thành số (0 và 1)
label_encoder = LabelEncoder()
data['Light Status'] = label_encoder.fit_transform(data['Light Status'])
data['Light Sensor Status'] = label_encoder.fit_transform(data['Light Sensor Status'])
# Chuyển đổi Day of the Week và Time thành các biến số
data['Day of the Week'] = label_encoder.fit_transform(data['Day of the Week'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour
# Tạo biến X (đặc trưng) và y (mục tiêu)
X = data[['Day of the Week', 'Time', 'Light Sensor Status']]
y = data['Light Status']
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Tiêu chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Bước 3: Xây dựng mô hình
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sử dụng 'sigmoid' cho bài toán phân loại nhị phân
])
# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Bước 4: Huấn luyện mô hình
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1)
# Bước 5: Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
# Bước 6: Lưu lại mô hình để sử dụng sau này
model.save('iot_model.keras')
print("Model saved")