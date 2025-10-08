"""
LINEAR REGRESSION - HỒI QUY TUYẾN TÍNH

- Linear Regression là một trong các thuật toán cơ bản nhất trong Machine Learning thuộc nhóm supervised learning
Hàm mô hình có dạng tuyến tính: y≈w1x1+w2x2+⋯+w0

Bài toán đặt ra: dự đoán cân nặng của một người dựa vào chiều cao
- có thể thấy là cân nặng sẽ tỉ lệ thuận với chiều cao (càng cao càng nặng), 
  nên có thể sử dụng Linear Regression model cho việc dự đoán này.
- 

"""

from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model 

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T # chiều cao (cm)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T # cân nặng (kg)


plt.plot(X, y, 'ro') 
plt.axis([140, 190, 45, 75]) # Đặt giới hạn cho trục x và trục y của biểu đồ.
plt.xlabel('Height (cm)') 
plt.ylabel('Weight (kg)') 
plt.show() 

# ---
# Phần 1: Xây dựng mô hình hồi quy tuyến tính từ đầu (from scratch) bằng NumPy
# ---

# Xây dựng ma trận Xbar để tính toán hệ số tự do (bias).
# Thêm một cột chứa các giá trị 1 vào ma trận X.
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Tính toán các trọng số (weights) của đường thẳng hồi quy
# Sử dụng công thức Normal Equation: w = (X^T * X)^-1 * X^T * y
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

# Chuẩn bị dữ liệu để vẽ đường thẳng hồi quy
# Lấy các giá trị w_0 và w_1 từ mảng w đã tính.
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2) # Tạo 2 điểm trên trục x để vẽ đường thẳng.
y0 = w_0 + w_1*x0 # Tính giá trị y tương ứng dựa trên công thức đường thẳng: y = w_0 + w_1*x.


plt.plot(X.T, y.T, 'ro') 
plt.plot(x0, y0) 
plt.axis([140, 190, 45, 75]) # Đặt lại giới hạn trục.
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show() 

y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )

# ---
# Phần 2: Sử dụng thư viện scikit-learn để so sánh
# ---

regr = linear_model.LinearRegression(fit_intercept=False) 
regr.fit(Xbar, y)

print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by manual calculation: ', w.T)