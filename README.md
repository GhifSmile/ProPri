# **Laporan Proyek Machine Learning - Ghifari Ismail**

## **Domain Proyek**
Dahulu pada zaman prasejarah, manusia purba menggunakan babi lu tina informasi yang terletak pada dinding-dinding gua sebagai [alat komunikasi](https://tirto.id/sejarah-perkembangan-teknologi-informasi-dan-komunikasi-gaJs). Mereka melukis pengalaman-pengalaman empiris yang mereka dapatkan pada dinding gua, seperti lukisan berburu dan lukisan berkebun untuk menyampaikan informasi. Di masa sejarah, alat komunikasi berevolusi menjadi lebih baik. Meskipun masih banyak yang menggunakan lukisan di dinding gua, hal tersebut dioptimalkan dengan penggunaan simbol-simbol seperti pada masa kehidupan Bangsa Sumeria yang memakai huruf piktograf di era 3000 SM. Pada masa modern, alat komunikasi semakin jauh berkembang. Kita bisa mendapatkan informasi mengenai hal apa pun, berbicara dengan siapa pun tanpa peduli akan jarak, dan berbagai macam hal lain dengan hanya menggunakan satu alat komunikasi yaitu handphone. Dewasa ini, banyak perusahaan-perusahaan pembuat handphone mengembangkan produknya agar menjadi lebih canggih, efektif, dan mungkin dapat menguasai pasar. Handphone dengan penambahan beberapa fitur canggih tersebut kini dinamai smartphone. Semakin banyaknya perusahaan-perusahaan pembuat produk handphone atau smartphone, membuat produsen serta konsumen harus dapat menyesuaikan produk terhadap finansial. Produsen harus dapat menyesuaikan harga produk berdasarkan fitur-fitur yang ada pada produknya dengan juga memperhatikan kondisi ekonomi target penjualan, sedangkan konsumen harus menyesuaikan harga produk terhadap spesifikasi produk dan finansialnya agar lebih optimal dalam penggunaan produk. Machine learning dapat membantu produsen maupun konsumen untuk melakukan optimalisasi penyesuaian produk terhadap finansial. Dalam proyek ini, ada beberapa tahapan sistematis yang harus dilalui dalam membuat produk machine learning. Tahapan tersebut dimulai dari Business Understanding sampai ke Model Evaluation.

## **Business Understanding**
Seperti yang sudah disampaikan sebelumnya, baik produsen maupun konsumen sangat ingin mendapatkan optimalisasi produk terhadap finansial. Produsen ingin mendapatkan pejualan yang seuntung-untungnya dengan tetap memerhatikan kondisi finansial target penjualan, serta melakukan optimalisasi harga terhadap produk agar tidak jauh dari harga pasaran. Lain halnya dengan produsen, konsumen ingin sekali mendapatkan keuntungan dengan menyesuaikan harga produk terhadap spesifikasi produk dan finansialnya agar lebih optimal dalam penggunaan produk.

### **Problem Statements**
Berdasarkan kondisi yang telah diuraikan sebelumnya, mengembangkan sebuah sistem klasifikasi rentang harga didasari oleh permasalahan berikut.

- Dari serangkaian variabel yang ada, variabel apa yang paling penting dalam melakukan klasifikasi rentang harga?
- Masuk ke dalam rentang harga apa suatu handphone dengan spesifikasi atau variabel tertentu?

### **Goals**
Berdasarkan kondisi yang telah diuraikan sebelumnya, mengembangkan sebuah sistem klasifikasi rentang harga mempunyai tujuan sebagai berikut.

- Membuat produsen dapat melakukan klasifikasi rentang harga handphone berdasarkan spesifikasi atau variabel yang ada.
- Membuat konsumen dapat menyesuaikan harga produk terhadap spesifikasi produk dan finansialnya agar lebih optimal dalam penggunaan produk.

### **Solution statements**
Berdasarkan kondisi serta permasalahan yang telah diuraikan sebelumnya, pada proyek ini akan dilakukan penyelesaian masalah yang relevan terhadap tujuan dengan mengembangkan suatu sistem klasifikasi rentang harga berdasarkan tiga algoritma machine learning. Ketiga algoritma tersebut akan dijelaskan sebagai berikut.

- **K-Nearest Neighbor**

K-Nearest Neighbor atau biasa disingkat KNN merupakan algoritma supervised
learning yang mana hasil dari instance yang baru diklasifikasikan berdasarkan mayoritas dari kategori k-tetangga terdekat. Algoritma ini bekerja dengan tahapan sebagai berikut:

1. Menghitung jarak
2. Menemukan tetangga terdekat
3. Menetapkan label

Kelebihan dari algoritma ini adalah cepatnya pelatihan dibandingkan algoritma-algoritma klasifikasi lainnya.

- **Random Forest**

Random forest merupakan salah satu metode dalam Decision Tree. Decision Tree atau pohon pengambil keputusan adalah sebuah diagram alir yang berbentuk seperti pohon yang memiliki sebuah root node yang digunakan untuk mengumpulkan data, Sebuah inner node yang berada pada root node yang berisi tentang pertanyaan tentang data dan  sebuah leaf node yang digunakan untuk memecahkan masalah serta membuat keputusan. Decision tree mengklasifikasikan suatu sampel data yang belum diketahui kelasnya kedalam kelas–kelas yang ada. Penggunaan decision tree agar dapat menghindari overfitting pada sebuah set data saat mencapai akurasi yang maksimum.

Random forest  adalah kombinasi dari  masing–masing tree yang baik kemudian dikombinasikan  ke dalam satu model. Random Forest bergantung pada sebuah nilai vector random dengan distribusi yang sama pada semua pohon yang masing masing decision tree memiliki kedalaman yang maksimal. Random forest adalah classifier yang terdiri dari classifier yang berbentuk pohon {h(x, θ k ), k = 1, . . .} dimana θk adalah random vector yang didistribusikan secara independen dan masing masing tree pada sebuah unit akan memilih class yang paling popular pada input x. Untuk lebih jelasnya mengenai algoritma random forest, silahkan kunjungi [tautan1](https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest/) atau [tautan2](https://www.datacamp.com/community/tutorials/random-forests-classifier-python).

- **AdaBoost**

Ada-boost atau Adaptive Boosting adalah salah satu ensemble boosting classifier yang diusulkan oleh Yoav Freund dan Robert Schapire pada tahun 1996. Ini menggabungkan beberapa classifier untuk meningkatkan akurasi classifier. AdaBoost adalah metode ensemble iteratif. Pengklasifikasi AdaBoost membangun pengklasifikasi yang kuat dengan menggabungkan beberapa pengklasifikasi berkinerja buruk sehingga Anda akan mendapatkan pengklasifikasi kuat dengan akurasi tinggi. Konsep dasar di balik Adaboost adalah mengatur bobot pengklasifikasi dan melatih sampel data di setiap iterasi sehingga memastikan prediksi akurat dari pengamatan yang tidak biasa. Setiap algoritma machine learning dapat digunakan sebagai pengklasifikasi dasar jika menerima bobot pada set pelatihan. Adaboost harus memenuhi dua syarat:

1. pengklasifikasi harus dilatih secara interaktif pada berbagai contoh pelatihan berbobot.
2. Dalam setiap iterasi, ia mencoba memberikan kecocokan yang sangat baik untuk contoh-contohnya dengan meminimalkan kesalahan pelatihan.

Untuk lebih jelasnya mengenai algoritma AdaBoost, silahkan kunjungi [AdaBoost Classifier in Python](https://www.datacamp.com/community/tutorials/adaboost-classifier-python).

## **Data Understanding**
Dataset yang akan digunakan untuk pengembangan model pada kasus ini adalah dataset yang diunduh dari kaggle yaitu [Mobile Price Classification](https://www.kaggle.com/iabhishekofficial/mobile-price-classification) dengan penggunaan penuh terhadap train datasetnya.

Untuk mengunduh dataset yang diperlukan dari kaggle lakukan tahapan-tahapan berikut:

1. Mengunduh kaggle.json dari API token pada profil kaggle
2. Upload file kaggle.json yang telah diunduh pada colab
3. Install library kaggle
4. Membuat direktori dengan nama ".kaggle"
5. Copy "kaggle.json" ke dalam direktori yang telah dibuat
6. Mengalokasikan izin yang diperlukan untuk file tersebut
7. Mendownload dataset mobile price classification

Untuk melakukan tahap ketiga sampai ketujuh tuliskan kode berikut.

```python
! pip install kaggle

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

# Mendownload dan menyiapkan dataset 
! kaggle datasets download iabhishekofficial/mobile-price-classification
```

Setelah dataset dapat diunduh, lakukan proses ekstraksi pada file dengan ekstensi zip agar dataset yang diperlukan dapat digunakan. Tuliskan kode berikut untuk proses ekstraksi.

```python
# Mengekstrak zip file
import zipfile

local_zip = '/content/mobile-price-classification.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()
```

Kemudian lakukan proses konversi dataset menjadi dataframe pada train.csv dan tampilkan lima sampel awal dari dataset pada dataframe. Tuliskan kode berikut untuk proses konversi serta menampilkan lima sampel awal.

```python
# Mengubah dataset menjadi dataframe
import pandas as pd

df = pd.read_csv('/content/train.csv')
df.head()
```

Deskripsi dari variabel-variabel yang ada pada dataset Mobile Price Classification adalah sebagai berikut:

- battery_power = energi total yang dapat disimpan baterai dalam satu waktu diukur dalam mAh
- blue = ada tidaknya bluetooth
- clock_speed = kecepatan di mana mikroprosesor mengeksekusi instruksi
- dual_sim = mendukung dual sim atau tidak
- fc = front camera dalam mega pixels
- four_g = mendukung 4G atau tidak
- int_memory = kapasitas internal memori dalam Gigabytes
- m_dep = kedalaman hp dalam cm
- mobile_wt = berat hp
- n_cores = jumlah cores dari processor
- pc = kamera utama dalam mega pixels
- px_height = tinggi resolusi pixel
- px_width = lebar resolusi pixel
- ram = jumlah ram dalam Mega Bytes
- sc_h = tinggi layar ponsel dalam cm
- sc_w = lebar layar ponsel dalam cm
- talk_time = waktu terlama satu kali pengisian baterai akan bertahan saat terakhir kali Anda berada
- three_g = mendukung 3G atau tidak
- touch_screen = mendukung layar sentuh atau tidak
- wifi = mendukung wifi atau tidak
- price_range = variabel target dengan nilai 0 (biaya rendah), 1 (biaya sedang), 2 (biaya tinggi) dan 3 (biaya sangat tinggi)

Pada tahap ini diperlukan juga analisis terhadap dataset untuk optimalisasi penggunaan yang mana langkah-langkahnya adalah sebagai berikut:

1. Melihat deskripsi statistik dataset dengan menuliskan kode

```python
# Melihat deskripsi statistik dataset
df.describe()
```

2. Memeriksa nilai-nilai pada variabel dengan melihat output dari langkah pertama untuk melihat ada atau tidaknya nilai yang tidak rasional.
3. Menghapus sampel yang memiliki nilai tidak rasional pada variabel.
4. Menangani Outliers dengan IQR Method dan menggunakan boxplot untuk visualisasi pada setiap variabel. Tuliskan kode berikut untuk visualisasi boxplot pada variabel yang ada.

```python
# Menangani Outliers dengan IQR Method dan menggunakan boxplot untuk visualisasi pada tiap variabel
import seaborn as sns

sns.boxplot(x=df['battery_power'])
```

![box_bat](https://github.com/GhifSmile/ProPri/blob/main/images/Capture_battery.PNG)

```python
sns.boxplot(x=df['clock_speed'])
```

![clck_spd](https://github.com/GhifSmile/ProPri/blob/main/images/Capture_clock_spd.PNG)

5. Melakukan drop pada sampel yang memiliki nilai outliers dengan menuliskan kode berikut.

```python
# Proses drop outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
```

6. Melakukan proses analisis data dengan teknik Univariate EDA dengan menuliskan kode sebagai berikut.

```python
# Melakukan proses analisis data dengan teknik Univariate EDA
import matplotlib.pyplot as plt
%matplotlib inline

df.hist(bins=50, figsize=(20,15))
plt.show()
```

![uni_eda](https://github.com/GhifSmile/ProPri/blob/main/images/Capture_UNI_EDA.PNG)

Dari histogram di atas, terdapat informasi bahwasanya pada tiap label/target price_range jumlahnya tidak berbeda secara signifikan, dengan kata lain, hal tersebut bisa dikatakan proporsional. Jumlah tertinggi ada pada target 3 (biaya sangat tinggi).


## **Data Preparation**
Pada tahap ini dilakukan beberapa optimalisasi terhadap penggunaan dataset yang ada. Hal tersebut dilakukan dengan menemukan variabel penting dengan scikit-learn menggunakan model random forest yang telah dilatih oleh train test sementara.

Tahapan-tahapan yang dilakukan adalah sebagai berikut:

1. Membuat model random forest yang dilatih oleh train test sementara dengan menuliskan kode sebagai berikut.

```python
# Menemukan variabel penting dengan scikit-learn menggunakan model random forest
from sklearn.model_selection import train_test_split

b = df['price_range'] 


li = list(df.columns.values[:20])
a = df[li]

# Split dataset menjadi train dan test
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=123)
```

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

clf=RandomForestClassifier(n_estimators=100)

clf.fit(a_train,b_train)

y_pred=clf.predict(a_test)


print("Accuracy:",metrics.accuracy_score(b_test, y_pred))
```

Dari model yang telah dibuat, didapat akurasi sebesar 0.8211678832116789 yang mana hal tersebut kurang mumpuni apabila digunakan.

2. Melihat akurasi dari tiap variabel dengan menuliskan kode

```python
var_imp = pd.Series(clf.feature_importances_,index=list(df.columns.values[:20])).sort_values(ascending=False)

var_imp
```

3. Membuat bar plot untuk visualisasi akurasi variabel dengan menuliskan kode berikut

```python
# Membuat bar plot
sns.barplot(x=var_imp, y=var_imp.index)
plt.xlabel('Skor variabel penting')
plt.ylabel('Variabel')
plt.title('Visualisasi variabel penting')
plt.legend()
plt.show()
```

![akur_skorVar_vis](https://github.com/GhifSmile/ProPri/blob/main/images/Capture_var_penting.PNG)

Terlihat bahwasanya ram merupakan variabel yang sangat penting. Kemudian ambil beberapa variabel penting lainnya dengan tolak ukur relatif berdasarkan spesifikasi, maka diambil beberapa variabel penting, yaitu ram, battery_power, px_width, px_height, mobile_wt, dan int_memory.

## **Modeling**
Pada tahap ini, akan dikembangkan model machine learning dengan tiga algoritma. Kemudian akan dievaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik.

Lakukan pembagian dataset terhadap variabel penting menjadi train test.

```python
from sklearn.model_selection import train_test_split

y = df['price_range'] 

# Ambil fitur penting
X = df[['px_width', 'px_height', 'battery_power', 'ram', 'mobile_wt', 'int_memory']]
# Split dataset menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```

Kemudian lalukan pelatihan dataset yang telah dibagi terhadap beberapa model dengan menuliskan kode sebagai berikut.

1. K-Nearest Neighbor dengan menuliskan
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
```

Pada model KNN diambil jumlah tetangganya bernilai 9. karena tidak ada metode terkhusus dalam memilih jumlah nilai tetangga, maka dilakukan metode intuisi dengan mengambil jumlah nilai tetangga ganjil terhadap kelas yang genap.

2. Random Forest dengan menuliskan
```python
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
```

Pada model Random Forest dilakukan optimalisasi sebelumnya dengan mengambil variabel-variabel penting.

3. AdaBoost dengan penggunaan svc base estimator, dengan menuliskan
```python
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

svc=SVC(probability=True, kernel='linear')

abc = AdaBoostClassifier(n_estimators=50,
                         base_estimator=svc,
                         learning_rate=1)

model = abc.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Pada model AdaBoost digunakan base_estimator svc untuk optimalisasi yang mana sebelumnya secara default menggunakan Decision Tree

## **Evaluation**
Pada tahap ini dilakukan evaluasi terhadap ketiga model yang telah dibuat dengan menggunakan [metrik akurasi](https://medium.com/@pararawendy19/memahami-metrik-pada-pemodelan-klasifikasi-29cd5b738ee7). Metrik akurasi adalah persentase jumlah data yang diprediksi secara benar terhadap jumlah keseluruhan data. Metrik akurasi digunakan pada kasus ini dikarenakan tiap label/target price_range jumlahnya tidak berbeda secara signifikan, dengan kata lain hal tersebut bisa dikatakan proporsional. Untuk penggunaannya, cukup menuliskan kode berikut pada tiap akhir pembuatan model

- KNN
```python
from sklearn import metrics

a_KNN = metrics.accuracy_score(y_test, y_pred)

print("Accuracy:",a_KNN)
```

Pada model KNN yang telah dibuat dan dilatih pada dataset yang sebelumnya telah dibagi, didapat akurasi sebesar 0.9197080291970803.

- Random forest
```python
from sklearn import metrics

a_RF = metrics.accuracy_score(y_test, y_pred)

print("Accuracy:",a_RF)
```

Pada model Random Forest yang telah dibuat dan dilatih pada dataset yang sebelumnya telah dibagi, didapat akurasi sebesar 0.8978102189781022.

- AdaBoot svc
```python
from sklearn import metrics

a_AdBoost = metrics.accuracy_score(y_test, y_pred)

print("Accuracy:",a_AdBoost)
```

Pada model AdaBoost dengan svc base estimator yang telah dibuat dan dilatih pada dataset yang sebelumnya telah dibagi, didapat akurasi sebesar 0.9635036496350365.

Tuliskan kode berikut untuk visualisasi model terhadap skor metrik akurasinya.

```python
kump_model = {'RF': a_RF, 'KNN': a_KNN, 'AdBoost_svc': a_AdBoost}

keys = kump_model.keys()

values = kump_model.values()

plt.bar(keys, values)

plt.title('Perbandingan akurasi model')

plt.ylabel('Score')
```

![akur_mod](https://github.com/GhifSmile/ProPri/blob/main/images/Capture_akurasi.PNG)

## **Kesimpulan**
Dari visualisasi yang ada terdapat interpretasi bahwasanya tiap model tidak terlalu berbeda secara signifikan terhadap skor akurasi. Namun, dari ketiga model yang ada, model AdaBoostlah yang paling optimal dengan memberikan skor akurasi di atas 95%. Maka pada kasus klasifikasi rentang harga handphone ini dapat disimpulkan bahwa penggunaan model AdaBoost sangat optimal.