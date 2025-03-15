import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. 数据读取与特征提取
def extract_features(file_path, target_peaks):
    """从CSV文件中提取目标特征峰的数据"""
    df = pd.read_csv(file_path, encoding='iso-8859-1', skiprows=15, header=None, names=["Pixel", "Raman Shift", "Raw", "Dark", "Dark Subtracted", "BaseLine Subtracted"])  # 跳过前16行元数据
    features = []
    for peak in target_peaks:
        # 找到最接近目标波数的行
        idx = (df['Raman Shift'] - peak).abs().idxmin()
        features.append(df.loc[idx, 'BaseLine Subtracted'])
    return features


# 定义目标特征峰的波数
target_peaks = [877.9292, 1048.8986, 1449.4435]

# 遍历所有浓度文件夹，提取特征和标签
features = []
labels = []
base_path = r"D:\桌面\预处理完成后拉曼数据"  # 替换为实际路径

for conc_folder in os.listdir(base_path):
    if "%浓度乙醇溶液" not in conc_folder:
        continue  # 跳过非浓度文件夹
    conc_str = conc_folder.split("%")[0]  # 提取百分号前的部分
    conc = int(conc_str)
    folder_path = os.path.join(base_path, conc_folder)

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            feat = extract_features(file_path, target_peaks)
            features.append(feat)
            labels.append(conc)

# 转换为数组
X = np.array(features)
y = np.array(labels)

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 在PCA步骤修改为保留2个主成分（即使方差不足95%）
pca = PCA(n_components=2)  # 强制保留2个主成分
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 绘制主成分得分图
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='tab20', alpha=0.6)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.title('PCA Score Plot of Gasoline Samples', fontsize=14)
plt.colorbar(scatter, label='Gasoline Class')
plt.grid(True)
plt.savefig('pca_score_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("保留的主成分数量:", pca.n_components_)


# 4. SVM模型训练与调参
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 'scale'],
    'kernel': ['rbf', 'linear']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=2)
grid_search.fit(X_train_pca, y_train)

best_svm = grid_search.best_estimator_
print("最佳参数:", grid_search.best_params_)

# 5. 模型评估
y_pred = best_svm.predict(X_test_pca)

print("\n测试集准确率:", accuracy_score(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns

# 生成混淆矩阵数据
cm = confusion_matrix(y_test, y_pred)

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


# 6. 新样本预测函数
def predict_new_sample(file_path, target_peaks, scaler, pca, model):
    """预测单个新样本的浓度"""
    # 提取特征
    features = extract_features(file_path, target_peaks)
    # 转换为数组并标准化
    X_new = np.array([features])
    X_new_scaled = scaler.transform(X_new)  # 使用训练时的scaler
    # PCA降维
    X_new_pca = pca.transform(X_new_scaled)
    # 预测
    prediction = model.predict(X_new_pca)
    return prediction[0]


# 7. 示例：预测单个文件
if __name__ == "__main__":
    # 替换为实际新样本路径
    new_file_path = r"D:\桌面\60度.csv"

    # 调用预测函数
    predicted_concentration = predict_new_sample(
        file_path=new_file_path,
        target_peaks=target_peaks,
        scaler=scaler,  # 使用训练时的标准化器
        pca=pca,  # 使用训练时的PCA模型
        model=best_svm  # 使用训练好的SVM模型
    )

    print(f"\n预测结果：该样本的乙醇浓度为 {predicted_concentration}%")
