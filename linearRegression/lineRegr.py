import numpy as np
from sklearn.linear_model import LinearRegression

# 准备数据
X = np.array([1, 2, 3, 4]).reshape(-1, 1)  # 特征变量 - 房屋面积
y = np.array([2, 3, 4, 5])  # 目标变量 - 房价

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X, y)

# 获取训练得到的参数
a = model.coef_[0]
b = model.intercept_

print(f"线性回归方程为: y = {a} * x + {b}")

# 预测新数据点
new_data = np.array([5]).reshape(-1, 1)
prediction = model.predict(new_data)

print(f"预测房屋面积为 5 时的房价: {prediction[0]}")
