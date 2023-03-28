import numpy as np

# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 损失函数
def loss(y, h):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

# 梯度下降法更新权重和偏置项
def gradient_descent(X, y, w, b, learning_rate):
    m = len(y)
    h = sigmoid(np.dot(X, w) + b)
    
    dw = np.dot(X.T, (h - y)) / m
    db = np.sum(h - y) / m
    
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    return w, b

# 训练逻辑回归模型
def train(X, y, iterations, learning_rate):
    w = np.zeros(X.shape[1])
    b = 0
    
    for _ in range(iterations):
        w, b = gradient_descent(X, y, w, b, learning_rate)
        
        # 打印损失函数值
        h = sigmoid(np.dot(X, w) + b)
        #print('Loss:', loss(y, h))
    
    return w, b

# 预测函数
def predict(X, w, b):
    return (sigmoid(np.dot(X, w) + b) > 0.5).astype(int)

# 示例数据
X = np.array([[30], [50], [70], [90]])
y = np.array([0, 0, 1, 1])

# 训练模型
iterations = 10000
learning_rate = 0.001
w, b = train(X, y, iterations, learning_rate)

# 预测新数据
X_new = np.array([[40], [80]])
y_pred = predict(X_new, w, b)
print('Predictions:', y_pred)
