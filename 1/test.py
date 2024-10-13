import pandas as pd
import tensorflow as tf

# Bước 1: Tải lại mô hình đã lưu
model = tf.keras.models.load_model('iot_model.h5')

# Bước 2: Nhận đầu vào từ người dùng
time_input = input("Nhập thời gian (hh:mm): ")
temperature = float(input("Nhập nhiệt độ (°C): "))
light = float(input("Nhập độ sáng (Lux): "))
motion_detected = int(input("Có chuyển động không (1 - Có, 0 - Không): "))
user_home = int(input("Người dùng có ở nhà không (1 - Có, 0 - Không): "))

# Chuyển đổi thời gian thành giờ và phút
time_parts = time_input.split(':')
hour = int(time_parts[0])
minute = int(time_parts[1])

# Tạo DataFrame cho đầu vào với 6 đặc trưng
input_data = pd.DataFrame({
    'Temperature (°C)': [temperature],
    'Light (Lux)': [light],
    'Motion Detected': [motion_detected],
    'User Home': [user_home],
    'Hour': [hour],  # Thêm giờ
    'Minute': [minute]  # Thêm phút
})

# Bước 3: Dự đoán các hoạt động còn lại
predictions = model.predict(input_data)[0]

# Chuyển đổi dự đoán thành trạng thái bật/tắt
light_pred = 1 if predictions[0] >= 0.5 else 0
ac_pred = 1 if predictions[1] >= 0.5 else 0
security_pred = 1 if predictions[2] >= 0.5 else 0

# Bước 4: Hiển thị kết quả dự đoán
print("\nKết quả dự đoán:")
print(f"Trạng thái đèn: {'Bật' if light_pred == 1 else 'Tắt'}")
print(f"Trạng thái điều hòa: {'Bật' if ac_pred == 1 else 'Tắt'}")
print(f"Trạng thái an ninh: {'Bật' if security_pred == 1 else 'Tắt'}")
