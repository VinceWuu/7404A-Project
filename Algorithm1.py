import numpy as np

rank_k_penalty = 0  # 初始化rank_k_penalty为0
lambda_2 = 0        # 初始化lambda_2为0
# 假设损失函数为平方损失
def loss_function(W, x):
    return np.dot(W, x) ** 2

# 预测函数为逻辑回归
def predict_sample(x, w):
    return 1 / (1 + np.exp(-np.dot(w, x)))

# # 计算G矩阵
def calculate_G(W, Xs):
    predictions = np.dot(Xs, W)  # 计算所有样本对所有分类器的预测值
    return predictions  # 返回预测矩阵G


def initialize_W(X_pos, X_neg, C1, C2):
    n_features = X_pos.shape[1]  # Number of features
    W0 = np.random.rand(n_features)  # Initialize W0 with random values

    W0_norm = np.linalg.norm(W0)  # L2 norm of W0
    loss_pos = np.sum([loss_function(W0, x) for x in X_pos])  # Loss on positive samples
    loss_neg = np.sum([np.sum([loss_function(W0, x) for x in X_neg])])  # Loss on negative samples

    W0 *= (W0_norm ** 2 / (loss_pos + C1 * loss_pos + C2 * loss_neg))

    return W0

def update_F_given_W(G, rank_k_penalty, lambda_2):
    U, s, Vt = np.linalg.svd(G, full_matrices=False)  # Singular Value Decomposition
    D_s = np.diag(np.maximum(s - rank_k_penalty, 0))  # Soft thresholding
    F = np.dot(U, np.dot(D_s, Vt))  # Reconstruct F
    return F

def update_W(X_pos, X_neg, W, F, C1, C2, learning_rate):
    P_pos = np.diag(np.array([predict_sample(x, w) for x, w in zip(X_pos, W)]))  # Positive sample predictions
    P_neg = np.array([[predict_sample(x, w) for w in W] for x in X_neg])  # Negative sample predictions
    I = np.eye(W.shape[1])
    ones = np.ones(W.shape[1])

    # Compute gradients
    grad_J = 2 * W + C1 * np.dot(X_pos.T, np.dot(P_pos - I, ones[:, np.newaxis])) + C2 * np.dot(X_neg.T, P_neg)
    G_W = np.dot(W, F) - F
    grad_H = 2 * np.dot(X_pos.T, G_W) - 2 * np.dot(np.dot(G_W, G_W.T), G_W - F)

    # Update W using gradients and learning rate
    W -= learning_rate * (grad_J + grad_H)

    return W

def LRE_SVMs(X_pos, X_neg, C1, C2, max_iterations=100):
    W = initialize_W(X_pos, X_neg, C1, C2)  # Step 1: Initialize W using equation (2)

    for _ in range(max_iterations):
        prev_obj_value = W
        G = calculate_G(W, np.vstack((X_pos, X_neg)))  # Step 3: Calculate the prediction matrix G(W)
        F = update_F_given_W(G, rank_k_penalty, lambda_2)  # Step 4: Solve for F using SVT method
        W = update_W(X_pos, X_neg, W, F, C1, C2, learning_rate=0.01)  # Step 5: Update W using gradient descent method

        current_obj_value = W

        converged = check_convergence(prev_obj_value, current_obj_value, tolerance=1e-5)  # 判断是否满足收敛条件
        if converged:
            print("Algorithm converged.")
            break  # 算法达到收敛条件，跳出循环
        else:
            prev_obj_value = current_obj_value  # 更新上一次迭代的目标函数值

    if not converged:
        print("Maximum number of iterations reached without convergence.")
    return W


def check_convergence(prev_obj_value, current_obj_value, tolerance=1e-5):
    obj_change = abs(prev_obj_value - current_obj_value)  # 计算目标函数值的变化
    if obj_change < tolerance:
        return True  # 目标函数值的变化小于阈值，算法收敛
    else:
        return False  # 目标函数值的变化仍大于阈值，算法尚未收敛
