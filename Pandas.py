import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = sns.load_dataset("titanic")
df.head()
df.shape

# Titanic veri setindeki erkek ve kadın yolcu sayıları:
df["sex"].value_counts()

# Her bir sütuna ait unique değerlerin sayısı ve pclass değişkeninin unique değerleri:
df.nunique()
df["pclass"].unique()

# pclass ve parch değişkenlerinin unique değerlerinin sayısı:
df[["pclass", "parch"]].nunique()

#embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz.
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()

# embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"] == "C"].head(10)

# embarked değeri S olmayanların tüm bilgelerini gösteriniz
df[df["embarked"] != "S"].head(10)

# Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df[(df["age"]<30) & (df["sex"]=="female")].head()

# Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
df[(df["fare"] > 500 ) | (df["age"] > 70 )].head()

# Her bir değişkendeki boş değerlerin toplamını bulunuz.
df.isnull().sum()

# who değişkenini dataframe'den düşürün.
df.drop("who", axis=1, inplace=True)

# deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()


# age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
df["age"].fillna(df["age"].median(),inplace=True)
df.isnull().sum()

# survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})


# 30 yaşının altında olanlara 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanic veri setinde age_flag adında bir değişken oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

def age_30(age):
    if age<30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))
df["age_flag"] = df["age"].apply(lambda x: 1 if x<30 else 0)


# Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
df = sns.load_dataset("tips")
df.head()
df.shape

# Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve
# ortalamasını bulunuz.
df.groupby("time").agg({"total_bill": ["sum","min","mean","max"]})

# Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df.groupby(["day","time"]).agg({"total_bill": ["sum","min","mean","max"]})

# Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve
# ortalamasını bulunuz.
df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                           "tip":  ["sum","min","max","mean"],
                                                                            "Lunch" : lambda x:  x.nunqiue()})

# size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalamasını bulun.
df.loc[(df["size"] < 3) & (df["total_bill"] >10 ) , "total_bill"].mean() # 17.184965034965035

# total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve
# tip in toplamını versin.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

# total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape
