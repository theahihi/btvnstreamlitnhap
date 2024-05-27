import streamlit as st
import pickle
import numpy as np

# Load the trained KNN model
model_path = 'knn_model4.pkl'
knn = pickle.load(open(model_path, 'rb'))

# Title of the application
st.title('Dự đoán khách hàng rời bỏ ngân hàng')

# Sidebar for user input
st.sidebar.title('Nhập các thuộc tính khách hàng')

# Number inputs for the features


id=st.number_input("id khách",0,82516,1002)
Age = st.number_input("Tuổi", 18, 42, 30)
CreditScore = st.number_input("Điểm tín dụng", 0, 656, 600)
Balance = st.number_input("Số dư tài khoản", 0.0, 55478.086689, 50000.0)
EstimatedSalary = st.number_input("Lương ước tính", 0.0, 112574.822734, 53213.0)

# Make predictions
input_data = np.array([[id,Age,CreditScore,Balance,EstimatedSalary]])
prediction = knn.predict(input_data)

# Display prediction
st.write('## Kết quả dự đoán:')
st.write('Khách hàng sẽ rời bỏ ngân hàng' if prediction[0] == 1 else 'Khách hàng sẽ không rời bỏ ngân hàng')
