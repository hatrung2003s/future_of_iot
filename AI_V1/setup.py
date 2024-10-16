import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv('1.csv')

# Bước 2: Chuyển đổi cột thời gian thành dạng datetime
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

# Bước 3: Tách các đặc trưng (features) và nhãn (labels)
X = df[['Temperature (°C)', 'Light (Lux)', 'Motion Detected', 'User Home']]
y = df[['Light Active', 'Air Conditioning Active', 'Security']]

# Bước 4: Thêm các thành phần thời gian (giờ và phút) vào đặc trưng
X['Hour'] = df['Time'].dt.hour  # Thêm giờ
X['Minute'] = df['Time'].dt.minute  # Thêm phút

# Bước 5: Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bước 6: Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Bước 7: Xây dựng mô hình với TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Cập nhật số lượng đặc trưng
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    # Ba đầu ra với activation sigmoid cho bài toán nhị phân
    tf.keras.layers.Dense(3, activation='sigmoid')
])

# Bước 8: Compile mô hình
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Bước 9: Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Bước 10: Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Bước 11: Lưu lại mô hình để sử dụng sau này
model.save('iot_model.h5')
print("Model saved as iot_model.h5")