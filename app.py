import streamlit as st
import pickle
import numpy as np

# Load the trained KNN model
model_path = 'bank_churn.pkl'
knn = pickle.load(open(model_path, 'rb'))

# Title of the application
st.title('Dự đoán khách hàng rời bỏ ngân hàng')

# Sidebar for user input
st.sidebar.title('Nhập các thuộc tính khách hàng')

# Number inputs for the features




customer_id = st.sidebar.number_input('Mã khách hàng', min_value=0, step=1)
credit_score = st.sidebar.number_input('Điểm tín dụng', min_value=0, max_value=900, 300,step=1)
age = st.sidebar.number_input('Tuổi', min_value=18, max_value=100, step=1)
balance = st.sidebar.number_input('Số dư tài khoản', min_value=0.0,5000000,step=1000.0, format="%.2f")
estimated_salary = st.sidebar.number_input('Lương ước tính', min_value=0.0, 10000000 ,step=1000.0, format="%.2f")

# Make predictions
input_data = np.array([[customer_id,age,credit_score,balance,estimated_salary]])
prediction = knn.predict(input_data)

# Display prediction
st.write('## Kết quả dự đoán:')
st.write('Khách hàng sẽ rời bỏ ngân hàng' if prediction[0] == 1 else 'Khách hàng sẽ không rời bỏ ngân hàng')
