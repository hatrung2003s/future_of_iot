import pandas as pd
import joblib

# Tải mô hình từ file
model = joblib.load('light_model.pkl')
print("Mô hình đã được tải thành công.")

# Nhập dữ liệu từ người dùng
light_sensor_status = int(input("Nhập trạng thái cảm biến ánh sáng (1: On, 0: Off): "))
time_input = input("Nhập thời gian (định dạng HH:MM): ")
day_of_week = int(input("Nhập ngày trong tuần (0: Monday, 1: Tuesday, ..., 6: Sunday): "))

# Tiền xử lý dữ liệu
hour, minute = map(int, time_input.split(':'))
data = pd.DataFrame({
    'Light Sensor Status': [light_sensor_status],
    'Hour': [hour],
    'Minute': [minute],
    'DayOfWeek': [day_of_week]
})

# Dự đoán
predictions = model.predict(data)

# Hiển thị kết quả dự đoán
print(f"Trạng thái đèn được dự đoán: {'On' if predictions[0] == 1 else 'Off'}")
