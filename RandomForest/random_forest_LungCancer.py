import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

pd.options.mode.chained_assignment = None

# Membaca data
data = dataframe = pd.read_csv(r'd:\COLLEGE\SEMESTER 5\MACHINE LEARNING\PRAKTIKUM\machine_learning\python\UTS-MLPRAK-KEL2-SMT5\lungCancer.csv', delimiter=';')

# Seleksi kolom yang akan digunakan
data = dataframe[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']]

# Mengubah kolom 'LUNG_CANCER' dan 'GENDER' menjadi numerik
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 2, 'NO': 1})
data['GENDER'] = data['GENDER'].map({'M': 0, 'F': 1})

# Tampilkan data awal
print("data awal".center(75, "="))
print(data)
print("=".center(75, "="))

# Pengecekan missing value
print("Pengecekan missing value".center(75, "="))
print(data.isnull().sum())
print("=".center(75, "="))

# Deteksi dan tampilkan outlier
def detect_outlier(data, threshold=3):
    outliers = []
    mean = np.mean(data)
    std = np.std(data)
    
    for yy in data:
        z_score = (yy - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(yy)
    
    return outliers

outliers = {}
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    outliers[col] = detect_outlier(data[col])

for col, outlier_values in outliers.items():
    if outlier_values:
        print(f"Outlier pada kolom {col}: {outlier_values}")
    else:
        print(f"Tidak ada outlier pada kolom {col}")

print("=".center(75, "=")) 

# Handling Outlier
# Hapus baris yang mengandung outlier
for col, outlier_values in outliers.items():
    if outlier_values:
        data = data[~data[col].isin(outlier_values)]

# Cetak hasil setelah menghapus outlier
for col, outlier_values in outliers.items():
    if outlier_values:
        print(f"Outlier pada kolom {col}: {outlier_values} telah dihapus.")
    else:
        print(f"Tidak ada outlier pada kolom {col}")

print("=".center(75, "="))

# Tampilkan data setelah handling outlier
print("data setelah handling outlier".center(75, "="))
print(data)
print("=".center(75, "="))

# Menyimpan data setelah handling outlier
output_file = r'D:\COLLEGE\SEMESTER 5\MACHINE LEARNING\PRAKTIKUM\machine_learning\python\UTS-MLPRAK-KEL2-SMT5\data_cleaned.csv'
data.to_csv(output_file, index=False)
print(f"Data setelah handling outlier disimpan ke '{output_file}'")

# Normalisasi data metode z-score standarisasi
standard_scaler = preprocessing.StandardScaler()
np_scaled = standard_scaler.fit_transform(data.drop(columns=['LUNG_CANCER']))
standardized = pd.DataFrame(np_scaled, columns=data.drop(columns=['LUNG_CANCER']).columns)
standardized['LUNG_CANCER'] = data['LUNG_CANCER'].values

print('\nData yang telah dinormalisasi dengan metode z-score standarisasi:')
print(standardized)

# Menyimpan data yang telah dinormalisasi ke file CSV
output_file = r'D:\COLLEGE\SEMESTER 5\MACHINE LEARNING\PRAKTIKUM\machine_learning\python\UTS-MLPRAK-KEL2-SMT5\data_normalized.csv'
standardized.to_csv(output_file, index=False)
print(f"Data yang telah dinormalisasi disimpan ke '{output_file}'")

# Grouping variabel untuk fitur dan label
print("GROUPING VARIABEL".center(75, "="))
X = standardized[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']].values  
y = data['LUNG_CANCER'].values  # Label
print("data variabel".center(75, "="))
print(X)
print("data kelas".center(75, "="))
print(y)
print("============================================================")

# Pembagian training dan testing
print("SPLITTING DATA 20-80".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("instance variabel data training".center(75, "="))
print(X_train)
print("instance kelas data training".center(75, "="))
print(y_train)
print("instance variabel data testing".center(75, "="))
print(X_test)
print("instance kelas data testing".center(75, "="))
print(y_test)
print("============================================================")
print()

# permodelan Random Forest
print("PERMODELAN DENGAN RANDOM FOREST".center(75, "="))
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
accuracy_rf = round(accuracy_score(y_test, Y_pred) * 100, 2)
print("instance prediksi random forest : ")
print(Y_pred)

# perhitungan confusion matrix algoritma random forest
cm_rf = confusion_matrix(y_test, Y_pred)
print("CLASSIFICATION REPORT RANDOM FOREST".center(75, "="))

# hasil akurasi
accuracy_rf = accuracy_score(y_test, Y_pred)
precision_rf = precision_score(y_test, Y_pred)
print(classification_report(y_test, Y_pred))

TN = cm_rf[1][1] * 1.0
FN = cm_rf[1][0] * 1.0
TP = cm_rf[0][0] * 1.0
FP = cm_rf[0][1] * 1.0
total = TN + FN + TP + FP
sens = TN / (TN + FP) * 100
spec = TP / (TP + FN) * 100

print("Akurasi algoritma : ", accuracy_rf * 100, "%")
print("Sensitivity algoritma : " + str(sens))
print("Specificity algoritma : " + str(spec))
print("Precission algoritma : " + str(precision_rf))

print("============================================================")
print()

# hasil confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
print("Hasil confusion matrix algoritma Random Forest\n", cm_rf)
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, Y_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("============================================================")
print()