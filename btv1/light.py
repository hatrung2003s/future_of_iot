import RPi.GPIO as GPIO
import time
import pandas as pd
import joblib
from datetime import datetime

# Thiết lập các chân GPIO
LIGHT_PIN = 12  # Chân GPIO để điều khiển đèn
LIGHT_SENSOR_PIN = 17  # Chân GPIO cho cảm biến ánh sáng

# Thiết lập GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(LIGHT_PIN, GPIO.OUT)
GPIO.setup(LIGHT_SENSOR_PIN, GPIO.IN)

# Tải mô hình từ file
model = joblib.load('light_model.pkl')
print("Mô hình đã được tải thành công.")

# Định nghĩa từ điển ánh xạ cho ngày trong tuần
day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

def read_light_sensor():
    """Đọc giá trị từ cảm biến ánh sáng"""
    return GPIO.input(LIGHT_SENSOR_PIN)

def control_light():
    """Điều khiển đèn dựa trên mô hình dự đoán"""
    while True:
        # Lấy thời gian hiện tại
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        day_of_week = current_time.strftime("%A")  # Lấy tên ngày trong tuần

        # Đọc trạng thái cảm biến ánh sáng
        light_sensor_status = read_light_sensor()
        light_sensor_status_input = 'On' if light_sensor_status == GPIO.HIGH else 'Off'

        # Tiền xử lý dữ liệu cho mô hình dự đoán
        data = {
            'Light Sensor Status': [light_sensor_status_input],
            'Hour': [current_hour],
            'Minute': [current_minute],
            'DayOfWeek': [day_mapping[day_of_week]]
        }
        
        input_data = pd.DataFrame(data)
        input_data['Light Sensor Status'] = input_data['Light Sensor Status'].map({'On': 1, 'Off': 0})

        try:
            # Dự đoán trạng thái đèn
            prediction = model.predict(input_data)

            # Điều khiển đèn dựa trên dự đoán
            if prediction[0] == 1:  # Bật đèn
                GPIO.output(LIGHT_PIN, GPIO.HIGH)
                light_status = "On"
            else:  # Tắt đèn
                GPIO.output(LIGHT_PIN, GPIO.LOW)
                light_status = "Off"

            # Hiển thị trạng thái của đèn, cảm biến và thời gian
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Trạng thái cảm biến ánh sáng: {light_sensor_status_input}")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Trạng thái đèn: {light_status}")

        except Exception as e:
            print(f"Lỗi trong quá trình dự đoán: {e}")

        # Đợi 5 giây trước khi kiểm tra lại
        time.sleep(5)

try:
    control_light()
except KeyboardInterrupt:
    print("Đang dừng chương trình...")
finally:
    GPIO.cleanup()  # Dọn dẹp GPIO khi thoát
