# :crocodile: GAT0R - Türkçe Doğal Dil İşleme Ekibi
---

<img  src="https://github.com/thealper2/gat0r-nlp/blob/main/flask/logo.jpg" alt="alt text" width="320" height="280">

*Bu çalışma, Teknofest 2023 Türkçe Doğal Dil İşleme yarışması kategorisinde "Aşağılayıcı Dil Tespiti" için geliştirilmiştir.*
### :crocodile: Amaç
---

Türkçe Doğal Dil İşleme ile özellikle Türkçe metinlerin işlenmesi için gerekli kullanıcı dostu ve yüksek performanslı kütüphanelerin, veri kümelerinin hazırlanmasına katkı sağlamak amaçlanmaktadır. Aşağılayıcı söylemlerin doğal dil işleme ile tespiti sağlanacaktır. Tespit edilen söylemler; SEXIST, INSULT, PROFANITY, RACIST ve OTHER olarak sınıflandırılacaktır. 

### :crocodile: Takım Üyeleri
---

1. [Alper KARACA](https://github.com/thealper2)
2. [Ferhat TOSON](https://github.com/ferhattoson)
3. [Aleyna KOCABEY](https://github.com/Leylaleyn)
4. [Fatma Begüm ARSLANOĞLU](https://github.com/FatmaBeg)


### :crocodile: Gerekli Kütüphaneler
---

* Çalışmalarımızın hepsi [Google Colab](https://colab.research.google.com/) ortamında yapılmıştır.

```shell
pandas==2.0
matplotlib==3.7.1
nltk==3.8.1
wordcloud==1.8.2.2
scikit-learnm==1.2.2
torch==2.0.0
transformers==4.27.4
numpy==1.24.2
tqdm==4.65.0
more-itertools==9.1.0
```

### :crocodile: Kodların Çalıştırılması

* Kodları aşağıdaki kodları yazarak çalıştırabilirsiniz.

```python
# Modeli çalıştırmak için
python3 bert_uncased-with-stopwords.py
# Gradio servisini çalıştırmak için
python3 gradio.py
# Flask servisini çalıştırmak için
python3 flask.py

```
* Gerekli kütüphaneleri kurmak için aşağıdaki kodu çalıştırabilirsiniz.

```python
pip3 install -r requirements.txt
```

# Model
---

* Modelimizi oluşturmak için aşağıdaki adımları uygulayacağız.

1. Veri setinin yüklenmesi ve incelenmesi
2. Veri ön işleme adımlarının uygulanması
3. Train, validation ve test veri sınıflarının oluşturulması ve model sınıfının oluşturulması
4. Model eğitme işlemi
5. Model doğrulama işlemi
6. Tahmin işleminin yapılması
7. Modelin kaydedilmesi

### :crocodile: Veri setini inceleyelim.
---
* Veri setine aşağıdaki linkden ulaşabilirsiniz. <br/>
[Veri Seti](https://raw.githubusercontent.com/thealper2/gat0r-nlp/main/datasets/teknofest_train.csv)
* Veri Setimizde 2081'i **SEXIST**, 2033'ü **RACIST**, 2376'sı **PROFANITY**, 2393'ü **INSULT**, 3584'ü **OTHER** kategorisinde olmak üzere12617 adet veri bulunmaktadır. 

| # |  Column   |Non-Null| Count|  Dtype |
|---|  ------ |  --------|------ | -----  |
| 0 | id      |12617 | non-null   |object  |
| 1 | text     |12617 |non-null   |object |
|2  | is_offensive|12617 |non-null |int64 |
|3  | target  |12617 |non-null |object|
 
<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/img1.png" alt="alt text" width="320" height="280">

* %28.7 si Other sınıfının içerisindeyken %71.3'ü Irkçı, Cinsiyetçi, Sexist,Küfür sınıfının içerisindedir.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/img2.png" alt="alt text" width="620" height="280" >

### :crocodile: Veri Ön İşleme
---
Aşağıdaki python kodu ile verisetimize ön işleme adımları uyguluyoruz.

```python
def preprocess_text(text):
    # Küçük harflere çevirme
    text = text.lower()
    # Satır sonu karakterlerini kaldırma
    import re
    text = re.sub(r'\n', ' ', text)
    # Rakamları kaldırma
    text = re.sub(r'\d', '', text)
    # Noktalama işaretlerini kaldırma
    import string
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    # Veri setindeki hatalı verilerin kaldırılması
    words = [word for word in words if not word in false_text]
    # Tekrarlanan karakterlerin kaldırılması
    words = [re.sub(r'(.)\1{1,}', r'\1\1', word) for word in words]
    # Tekrarlanan boşlukların kaldırılması
    words = [word.strip() for word in words if len(word.strip()) > 1]
    
    text = " ".join(words)
    return text
```

* Veri setini temizledikten sonra ön işleme adımlarından olan tokenize, mapping ve encoder dönüşümünü yapılmıştır.

```python
from sklearn.preprocessing import LabelEncoder
# LabelEncoder kullanarak "target" sütunumuza dönüşüm yaptırdık.
le = LabelEncoder().fit(df["target"])
# "Name Mapping" işlemini tanımladık.
le_nm = dict(zip(le.classes_, le.transform(le.classes_)))
# Veri setimizde "labels" adında sütun oluşturarak "target" sütununa "Name Mapping" işlemi yaparak "labels" sütunu altına aktardık.
df["labels"] = df["target"].apply(lambda x: le_nm[x])
# "id", "text", "target" sütunlarını veri setimizden çıkardık.
df = df.drop(['id', 'text', 'target'], axis=1)
!pip install transformers
```
* "Name Mapping" işleminden sonra veri setimizin ilk 5 elemanı inceleyelim.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/img0.png">

* LabelEncoder işleminden sonra her bir kategoriyi temsil eden sayılar;
- 0: INSULT
- 1: OTHER
- 2: PROFANITY
- 3: RACIST
- 4: SEXIST

### :crocodile: Train, validation ve test sınıfın oluşturulması
---

* Modelimizi eğitmek için veri setimizin **%60**'ın train, **%20**'sini validation ve **%20**'sini test işlemi için böldük.

```python
def train_validate_test_split(df):
    # Rastgelelik durumu.
    np.random.seed(4242)
    # Diziyi rastgele permute eder.
    perm = np.random.permutation(df.index)
    # Veri setinin %60'ının sayısal değeri hesaplandı.
    train_end = int(.6 * len(df.index))
    # Veri setinin %20'sinin sayısal değeri hesaplandı.
    validate_end = int(.2 * len(df.index)) + train_end
    # Veri setinin %60'ını train etmek için ayırdık.
    train = df.iloc[perm[:train_end]]
    # Veri setinin %20'sini validation etmek için ayırdık.
    validate = df.iloc[perm[train_end:validate_end]]
    # Veri setinin %20'sini test etmek için ayırdık.
    test = df.iloc[perm[validate_end:]]
    # train, validation, test veri setlerini döndür.
    return train, validate, test

# Train, Validation ve Test için veri setlerimiz oluşturduk.
df_train, df_validation, df_test = train_validate_test_split(df)
# Train, Validation ve Test için oluşturduğumuz veri setlerinin uzunluğunu ekrana yazdırdık. (%60 - %20 - %20)
print(len(df_train), len(df_validation), len(df_test))
```

### :crocodile: Model sınıfını oluşturmak
---

* BertClassifier'imizi oluşturmak için, 1 Bert, 1 Dropout, 1 Linear, 1 ReLU katmanı kullanıyoruz.


### :crocodile: Modeli eğitmek ve doğrulamak
---

* Modelimizi eğitmek için aşağıdaki parametreleri kullanıyoruz.
- EPOCHS = 2
- LEARNING RATE = 1e-6
- BATCH SIZE=2

* Eğitme ve doğrulama işleminden sonra aşağıdaki skorları elde ediyoruz.

| EPOCHS | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ------ | ---------- | -------------- | --------------- | ------------------- |
| 1 | 0.5284 | 0.6281 | 0.2232 | 0.8763 |
| 2 | 0.1590 | 0.9103 | 0.1416 | 0.9120 |


### :crocodile: Tahmin
---

* Tahmin işleminden sonra aşağıdaki skorları elde ediyoruz.

| # | precision | recall | f1-score | support |
| - | --------- | ------ | -------- | ------- |
| 0 | 0.86 | 0.86 | 0.86 | 489 |
| 1 | 0.94 | 0.91 | 0.93 | 718 |
| 2 | 0.93 | 0.92 | 0.93 | 477 |
| 3 | 0.89 | 0.94 | 0.92 | 399 | 
| 4 | 0.92 | 0.94 | 0.93 | 441 |
| accuracy | x | x | 0.91 | 2524 |
| macro avg | 0.91 | 0.91 | 0.91 | 2524 |
| weighted avg | 0.91 | 0.91 | 0.91 | 2524 |

* Modelimizin tahmin işleminden sonra oluşan sonucu görebilmek için **plot_confusion_matrix** fonksiyonumuzu kullanıyoruz.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/confmatrix.png" alt="confusion matrix">

### :crocodile: Modeli Kaydedelim
---

* Oluşan modelimizi "pt" uzantısı ile **pytorch** modeli halinde Google Drive'ımıza kaydediyoruz.

```python
from google.colab import drive
drive.mount('/content/gdrive')

# Modelimize isim vererek Drive'a kaydettik.
model_name = "bert_uncased-with-stopwords.pt"
path = F"/content/gdrive/My Drive/{model_name}"

torch.save(model.state_dict(), path)
```


### :crocodile: Modeller
---

Proje gelişimi boyunca, BERTurk'ün *cased* ve *uncased* versiyonları kullanılmıştır. Bu versiyonların üzerinde etkisiz sözcüklerin etkileri ortaya çıkmıştır. Kullanılan modelin HuggingFace sayfasına [buradan](https://huggingface.co/dbmdz/bert-base-turkish-uncased) ulaşabilirsiniz.


|# | F1-score | Time |
|------------|-------|------|
|BERT-cased-with-stopwords| %89 |29:36 dk|
|BERT-cased-without-stopwords| %91|29:15 dk|
|BERT-uncased-without-stopwords| %92|28:63 dk|
|BERT-uncased-with-stopwords| %91| 28:64 dk|

* Projenin canlıya alınmasında **BERT-uncased-with-stopwords** modeli kullanılmıştır. Modellerin kodlarına gitmek için aşağıdaki linklere tıklayın.

| # | Python | Jupyter Notebook |
|---|--------|------------------|
| BERT-cased-with-stopwords | [Link](https://github.com/thealper2/gat0r-nlp/blob/main/models/bert_cased_with_stopwords.py) | [Link](https://github.com/thealper2/gat0r-nlp/blob/main/models/BERT_cased_with_stopwords.ipynb) |
| BERT-cased-without-stopwords | [Link](https://github.com/thealper2/gat0r-nlp/blob/main/models/bert_cased_without_stopwords.py) | [Link](https://github.com/thealper2/gat0r-nlp/blob/main/models/BERT_cased_without_stopwords.ipynb) |
| BERT-uncased-with-stopwords | [Link](https://github.com/thealper2/gat0r-nlp/blob/main/models/bert_uncased_with_stopwords.py) | [Link](https://github.com/thealper2/gat0r-nlp/blob/main/models/BERT_uncased_with_stopwords.ipynb) |
| BERT-uncased-without-stopwords | [Link](https://github.com/thealper2/gat0r-nlp/blob/main/models/bert_uncased_without_stopwords.py) | [Link](https://github.com/thealper2/gat0r-nlp/blob/main/models/BERT_uncased_without_stopwords.ipynb)) |

* Kaydedilmiş modellere ulaşmak için aşağıdaki linki kullanabilirsiniz.
[Modeller (Drive)](https://drive.google.com/drive/folders/1Wni5jOcrAp7GTONO5Sx079JFlXfYWQPQ?usp=sharing)

### :crocodile: GAT0R SEARCH
---

Flask kullanarak girilen metnin sınıflandırmasını yapan bir uygulama geliştirdik.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/flask-resim.png?raw=true" alt="flask uygulamasi">

### :crocodile: HuggingFace Spaces
---

Modelimizin Gradio servisine aşağıdaki linkden ulaşabilirsiniz.<br/>
[Gradio](https://huggingface.co/spaces/thealper2/gat0r-gradio)<br/>
[Kaynak Kodu](https://github.com/thealper2/gat0r-nlp/blob/main/NLPEvaluation_GAT0R.py)

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/huggingface-gradio.png?raw=true" alt="gradio">


### :crocodile: Kaynaklar
---

[Text Classification with BERT in PyTorch](https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f)

### :crocodile: Lisans
---

Lisans dosyasına [buradan](https://github.com/thealper2/gat0r-nlp/blob/main/LICENSE) ulaşabilirsiniz.

