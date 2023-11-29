import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, auc, roc_curve, \
    accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
# from tabulate import tabulate

import warnings as warn
from warnings import filterwarnings

filterwarnings("ignore")

data = pd.read_csv("./IRIS.csv")
df = pd.DataFrame(data)
# print(df)

grpbySpecies = df.groupby('species').count()
# print(grpbySpecies)

df["species"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}, inplace=True)
# print(df)

# print(df.describe(include = 'all'))

plt.figure(figsize=(16, 6))

mask = np.triu(np.ones_like(df.corr(method="spearman"), dtype=bool))
heatmap = sns.heatmap(df.corr(method="spearman"), mask=mask, vmin=-1, vmax=1, annot=True, cmap="BrBG")
heatmap.set_title("Triangle Correlation Heatmap", fontdict={'fontsize': 18}, pad=16)

sns.pairplot(df, hue='species', diag_kind="hist", corner=True, palette='hls')
# plt.show()

Num = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

j = 0
while j < 5:
    fig = plt.figure(figsize=[20, 4])
    plt.subplot(1, 2, 1)
    sns.boxplot(x=Num[j], data=df, color='skyblue')
    sns.set(font_scale=1.25)
    j += 1
    plt.subplot(1, 2, 2)
    sns.boxplot(x=Num[j], data=df, color='skyblue')
    sns.set(font_scale=1.25)
    j += 1
    if j == 4:
        break
    plt.show()

sns.countplot(x=df['species'], data=df)

X = pd.DataFrame(df, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
y = df["species"].values.reshape(-1, 1)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


def Evaluate_Performance(Model, Xtrain, Xtest, Ytrain, Ytest):
    Model.fit(Xtrain, Ytrain)
    overall_score = cross_val_score(Model, Xtrain, Ytrain, cv=10)
    model_score = np.average(overall_score)
    Ypredicted = Model.predict(Xtest)
    avg = 'weighted'
    print("\n • Training Accuracy Score : ", round(Model.score(Xtrain, Ytrain) * 100, 2))
    print(f" • Cross Validation Score : {round(model_score * 100, 2)}")
    print(f" • Testing Accuracy Score :{round(accuracy_score(Ytest, Ypredicted) * 100, 2)}")
    print(f" • Precision Score is : {np.round(precision_score(Ytest, Ypredicted, average=avg) * 100, 2)}")
    print(f" • Recall Score is : {np.round(recall_score(Ytest, Ypredicted, average=avg) * 100, 2)}")
    print(f" • F1-Score Score is : {np.round(f1_score(Ytest, Ypredicted, average=avg) * 100, 2)}")


# Khởi tạo các danh sách để lưu trữ độ chính xác trên tập huấn luyện và tập kiểm thử
training_acc = []  # Danh sách lưu độ chính xác trên tập huấn luyện
test_acc = []  # Danh sách lưu độ chính xác trên tập kiểm thử

# Xác định khoảng số lượng láng giềng (neighbors) từ 1 đến 29
neighbors_setting = range(1, 30)

# Duyệt qua các giá trị số láng giềng
for n_neighbors in neighbors_setting:
    # Khởi tạo mô hình KNN với số láng giềng được xác định
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Huấn luyện mô hình trên tập huấn luyện
    KNN.fit(X_train, y_train.ravel())

    # Đo độ chính xác trên tập huấn luyện và lưu vào danh sách
    training_acc.append(KNN.score(X_train, y_train))

    # Đo độ chính xác trên tập kiểm thử và lưu vào danh sách
    test_acc.append(KNN.score(X_test, y_test))

plt.plot(neighbors_setting, training_acc, label="Độ chính xác trên tập huấn luyện")
plt.plot(neighbors_setting, test_acc, label="Độ chính xác trên tập kiểm thử")
plt.xlabel("Số lượng láng giềng")
plt.ylabel("Độ chính xác")
plt.grid(linestyle='-')
plt.legend()

# Thiết lập các tham số cần tối ưu hóa cho mô hình KNN
parameters = {"n_neighbors": range(1, 50)}

# Khởi tạo đối tượng GridSearchCV để tìm kiếm siêu tham số tối ưu
grid_kn = GridSearchCV(estimator=KNN, param_grid=parameters, scoring="accuracy", cv=5, verbose=1, n_jobs=-1)

# Tiến hành tìm kiếm siêu tham số trên tập huấn luyện
grid_kn.fit(X_train, y_train.ravel())

# Trả về giá trị của siêu tham số tối ưu được chọn bởi GridSearchCV
grid_kn.best_params_

# Xác định số lượng láng giềng K trong mô hình K-Nearest Neighbors
K = 3

# Khởi tạo mô hình KNN với số láng giềng được xác định
KNN = KNeighborsClassifier(K)

# Huấn luyện mô hình trên tập huấn luyện
KNN.fit(X_train, y_train.ravel())

# Dự đoán nhãn cho tập kiểm thử sử dụng mô hình đã huấn luyện
y_pred_KNN = KNN.predict(X_test)

# In thông tin về mô hình và đánh giá hiệu suất
print("K-Nearest Neighbors : ")
Evaluate_Performance(KNN, X_train, X_test, y_train, y_test)

# Sử dụng phương pháp Cross-Validation để đánh giá hiệu suất của mô hình K-Nearest Neighbors
KNN_r = cross_val_score(KNN, X, y, cv=10)

# Tính độ lệch chuẩn (standard deviation) của các điểm đánh giá Cross-Validation
K = np.std(KNN_r)

# In ra giá trị độ lệch chuẩn, giúp đánh giá sự biến động của hiệu suất trên các fold khác nhau
print(K)

# Tính ma trận nhầm lẫn (confusion matrix) giữa các nhãn thực tế và dự đoán của mô hình K-Nearest
cm = confusion_matrix(y, KNN.predict(X))

# Hiển thị ma trận nhầm lẫn dưới dạng hình ảnh sử dụng đồ thị
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Nhãn dự đoán', fontsize=14, color='black')
ax.set_ylabel('Nhãn thực tế', fontsize=14, color='black')
ax.xaxis.set(ticks=range(3))
ax.yaxis.set(ticks=range(3))
ax.set_ylim(2.5, -0.5)

# Hiển thị số lượng mẫu của từng phần tử trong ma trận nhầm lẫn
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

# Hiển thị đồ thị
plt.show()

