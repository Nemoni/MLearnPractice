import numpy as np
import pandas as pd
from collections import Counter


def entropy(y):
    freqs = np.array(list(Counter(y).values())) / len(y)
    return -np.sum(freqs * np.log2(freqs))


def information_gain(X, y, feature):
    entropy_before = entropy(y)
    values, counts = np.unique(X[:, feature], return_counts=True)
    entropy_after = np.sum([(counts[i] / len(y)) * entropy(y[X[:, feature] == values[i]]) for i in range(len(values))])
    return entropy_before - entropy_after


def split(X, y, feature, value):
    indices = X[:, feature] == value
    return X[indices], y[indices]


def build_tree(X, y, features, max_depth=None, depth=0):
    if max_depth is not None and depth == max_depth or len(np.unique(y)) == 1:
        return Counter(y).most_common(1)[0][0]
    
    best_feature = max(features, key=lambda f: information_gain(X, y, f))
    remaining_features = features - {best_feature}
    
    tree = {}
    for value in np.unique(X[:, best_feature]):
        X_subset, y_subset = split(X, y, best_feature, value)
        
        if len(y_subset) == 0:
            tree[(best_feature, value)] = Counter(y).most_common(1)[0][0]
        else:
            tree[(best_feature, value)] = build_tree(X_subset, y_subset, remaining_features, max_depth, depth + 1)
    
    return tree


def predict(tree, x):
    for (feature, value), subtree in tree.items():
        if x[feature] == value:
            if isinstance(subtree, dict):
                return predict(subtree, x)
            else:
                return subtree
    return None


# Example dataset
data = pd.DataFrame({
    '天气': ['晴天', '晴天', '阴天', '阴天', '雨天'],
    '温度': ['炎热', '适中', '炎热', '寒冷', '适中'],
    '是否去公园玩耍': ['否', '是', '否', '否', '是']
})

X = data[['天气', '温度']].values
y = data['是否去公园玩耍'].values
features = set(range(X.shape[1]))

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
encoders = [LabelEncoder().fit(X[:, i]) for i in range(X.shape[1])]
X_encoded = np.column_stack([encoders[i].transform(X[:, i]) for i in range(X.shape[1])])

# Build decision tree
tree = build_tree(X_encoded, y, features)

# Predict
x_new = ['晴天', '炎热']
x_new_encoded = np.array([encoders[i].transform([x_new[i]])[0] for i in range(len(x_new))])
prediction = predict(tree, x_new_encoded)
print(f"预测结果：{prediction}")  # 预测结果：否
