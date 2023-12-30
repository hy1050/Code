import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load mô hình từ tệp .h5
model = load_model(r'C:\Users\BHG81HC\Documents\Private_Hy\Graduate_Thesis\Documents\Model_CNN\my_model_cnn.h5')

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons',
            43:'Do not turn leff',
            44:'Do not turn right',
            45:'Speed limit (40km/h)',
            46:'No parking and stopping',
            47:'Danergous bend ahead',
            48:'No parking beyond this point',
            49:'Maximum weights on axis 10T',
            50:'Intersection with non-priority road',
            51: 'Pedestrian Crossing',
            52: 'None'}


# Cấu hình camera (chú ý rằng các thông số này có thể thay đổi tùy thuộc vào loại camera)
cap = cv2.VideoCapture(0)  # 0 là ID của camera, có thể cần điều chỉnh nếu có nhiều camera
if not cap.isOpened():
    print('can not open video clip/camera')
    exit()
# Biến tốc độ hiện tại
current_speed = 59  # Điều chỉnh tốc độ hiện tại tùy thuộc vào tình huống thực tế

while True:
    # Chụp hình ảnh từ camera
    ret, frame = cap.read()

    # Xử lý hình ảnh để phù hợp với đầu vào của mô hình
    img = cv2.resize(frame, (32, 32))  # Điều chỉnh kích thước tùy thuộc vào kiến trúc mô hình của bạn
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    predictions = model.predict(img_array)

    # Xử lý kết quả dự đoán ở đây (ví dụ: in ra lớp có xác suất cao nhất)
    predicted_class = np.argmax(predictions[0])
    print("Predicted class:", predicted_class)

    # Thêm điều kiện để kiểm tra xem có phải là biển báo tốc độ không
    if predicted_class == your_speed_limit_sign_class_id:
        speed_limit = extract_speed_limit(predictions[0])  # Hàm giả định để trích xuất giới hạn tốc độ từ dự đoán
        if current_speed > speed_limit:
            print("Warning: Speed Limit Exceeded!")
            current_speed -= 5  # Giảm tốc độ hiện tại theo bước 5km/lần

    # Hiển thị hình ảnh với kết quả dự đoán
    cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera Feed', frame)

    # Thoát vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()