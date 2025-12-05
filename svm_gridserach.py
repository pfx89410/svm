import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. 32x32 文本图片 -> 1x1024 向量 ==========
def img2vector(file_path):
    """将32x32的0/1文本图片转成1x1024的numpy向量"""
    vec = np.zeros((1, 1024), dtype=np.float32)
    with open(file_path, 'r') as f:
        for i in range(32):
            line_str = f.readline().strip()
            line_str = line_str[:32].ljust(32, '0')  # 防御性处理
            for j in range(32):
                vec[0, 32 * i + j] = int(line_str[j])
    return vec

# ========== 2. 读取整个数据集 ==========
def load_dataset(dir_path):
    """遍历目录下txt文件，提取特征和标签（文件名格式：digit_index.txt）"""
    file_list = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    num_files = len(file_list)

    data_mat = np.zeros((num_files, 1024), dtype=np.float32)
    label_list = []

    for i, file_name in enumerate(file_list):
        full_path = os.path.join(dir_path, file_name)
        data_mat[i, :] = img2vector(full_path)
        # 提取标签（_前的数字）
        class_str = file_name.split('_')[0]
        label_list.append(int(class_str))

    return data_mat, np.array(label_list, dtype=np.int32)

# ========== 3. 加载数据集 ==========
train_dir = r"C:\Users\E507\Documents\GitHub\svm\dataset\trainingDigits"
test_dir  = r"C:\Users\E507\Documents\GitHub\svm\dataset\testDigits"

X_train, y_train = load_dataset(train_dir)
X_test,  y_test  = load_dataset(test_dir)

print("训练集形状：", X_train.shape, " 标签形状：", y_train.shape)
print("测试集形状：", X_test.shape,  " 标签形状：", y_test.shape)
print("="*50)

# ========== 4. SVM参数网格搜索 ==========
svc = SVC(kernel="rbf", random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("开始网格搜索最优SVM参数...")
grid_search.fit(X_train, y_train)

print("="*50)
print("最优参数组合：", grid_search.best_params_)
print("5折交叉验证最佳平均准确率：", round(grid_search.best_score_, 4))

# ========== 5. 测试集评估 ==========
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print("="*50)
print(f"测试集准确率：{test_acc:.4f}")

# ========== 修复核心：显式指定labels和动态target_names ==========
# 1. 提取唯一类别并排序（确保顺序固定）
unique_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
# 2. 生成对应类别名称
target_names = [f"数字{cls}" for cls in unique_classes]

print("="*50)
print("详细分类报告（按数字类别）：")
print(classification_report(
    y_test, 
    y_pred,
    labels=unique_classes,  # 显式指定类别顺序（关键修复）
    target_names=target_names,  # 动态匹配类别名称
    digits=4
))