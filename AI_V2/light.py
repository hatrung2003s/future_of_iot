import RPi.GPIO as GPIO
import time
import numpy as np
import tensorflow as tf

# Thiết lập GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)  # Chân 17 cho đèn
GPIO.setup(12, GPIO.IN)   # Chân 12 cho cảm biến

# Tải mô hình
model = tf.keras.models.load_model('iot_model.keras')

def predict_light_status(day_of_week, hour, light_sensor_status):
    # Tạo đầu vào cho mô hình
    input_data = np.array([[day_of_week, hour, light_sensor_status]])
    
    # Dự đoán trạng thái đèn
    prediction = model.predict(input_data)
    predicted_status = 1 if prediction[0][0] > 0.5 else 0  # Ngưỡng 0.5
    return predicted_status

try:
    while True:
        # Đọc dữ liệu từ cảm biến
        light_sensor_status = GPIO.input(12)  # Giả sử 0 là OFF và 1 là ON
        
        # Lấy thời gian hiện tại
        current_time = time.localtime()
        day_of_week = current_time.tm_wday  # Thứ trong tuần (0=Thứ Hai, 6=Chủ Nhật)
        hour = current_time.tm_hour  # Giờ hiện tại
        minute = current_time.tm_min  # Phút hiện tại
        
        # Dự đoán trạng thái đèn
        light_status = predict_light_status(day_of_week, hour, light_sensor_status)
        
        # Hiển thị thông tin trạng thái cảm biến và thời gian
        sensor_status_text = "ON" if light_sensor_status else "OFF"
        print(f"Thời gian: {hour:02d}:{minute:02d}, Trạng thái cảm biến: {sensor_status_text}")
        
        # Điều khiển đèn
        if light_status == 1:
            GPIO.output(17, GPIO.HIGH)  # Bật đèn
            print("Đèn đã được bật.")
        else:
            GPIO.output(17, GPIO.LOW)   # Tắt đèn
            print("Đèn đã được tắt.")
        
        time.sleep(10)  # Đọc lại sau mỗi 10 giây

except KeyboardInterrupt:
    print("Đã dừng chương trình.")

finally:
    GPIO.cleanup()  # Dọn dẹp GPIO
