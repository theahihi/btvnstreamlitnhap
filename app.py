import streamlit as st
import pickle
import numpy as np

# Load the trained KNN model
model_path = 'knn_model2.pkl'
knn = pickle.load(open(model_path, 'rb'))

# Title of the application
st.title('Dự đoán khách hàng rời bỏ ngân hàng')

# Sidebar for user input
st.sidebar.title('Nhập các thuộc tính khách hàng')

# Number inputs for the features

customer_id = st.sidebar.number_input('Mã khách hàng', min_value=0, step=1)
credit_score = st.sidebar.number_input('Điểm tín dụng', min_value=300, max_value=850, step=1)
geography = st.sidebar.selectbox('Quốc gia', ['France', 'Germany', 'Spain'])
gender = st.sidebar.selectbox('Giới tính', ['Nam', 'Nữ'])
age = st.sidebar.number_input('Tuổi', min_value=18, max_value=100, step=1)
tenure = st.sidebar.number_input('Thời gian sử dụng dịch vụ (năm)', min_value=0, max_value=10, step=1)
balance = st.sidebar.number_input('Số dư tài khoản', min_value=0.0, step=1000.0, format="%.2f")
num_of_products = st.sidebar.number_input('Số lượng sản phẩm sử dụng', min_value=1, max_value=4, step=1)
has_cr_card = st.sidebar.selectbox('Có thẻ tín dụng', [0, 1])
is_active_member = st.sidebar.selectbox('Là thành viên tích cực', [0, 1])
estimated_salary = st.sidebar.number_input('Lương ước tính', min_value=0.0, step=1000.0, format="%.2f")

# Convert inputs to appropriate format
geography_dict = {'France': 0, 'Germany': 1, 'Spain': 2}
gender_dict = {'Nam': 1, 'Nữ': 0}

geography = geography_dict[geography]
gender = gender_dict[gender]

# Make predictions
input_data = np.array([[customer_id, credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])
prediction = knn.predict(input_data)

# Display prediction
st.write('## Kết quả dự đoán:')
st.write('Khách hàng sẽ rời bỏ ngân hàng' if prediction[0] == 1 else 'Khách hàng sẽ không rời bỏ ngân hàng')
