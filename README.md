# GAT0R - Türkçe Doğal Dil İşleme Ekibi
---

<img  src="https://github.com/thealper2/gat0r-nlp/blob/main/flask/logo.jpg" alt="alt text" width="320" height="280">

*Bu çalışma, Teknofest 2023 Türkçe Doğal Dil İşleme yarışması kategorisinde "Aşağılayıcı Dil Tespiti" için geliştirilmiş. *
### Amaç

Türkçe Doğal Dil İşleme ile özellikle Türkçe metinlerin işlenmesi için gerekli kullanıcı dostu ve yüksek performanslı kütüphanelerin, veri kümelerinin hazırlanmasına katkı sağlamak amaçlanmaktadır. Aşağılayıcı Söylemlerin Doğal Dil İşleme İle Tespiti sağlanacaktır. 
### Veri setini inceleyelim.

| # |  Column   |Non-Null| Count|  Dtype |
|---|  ------ |  --------|------ | -----  |
| 0 | id      |12617 | non-null   |object  |
| 1 | text     |12617 |non-null   |object |
|2  | is_offensive|12617 |non-null |int64 |
|3  | target  |12617 |non-null |object|

 Veri Setimizde 12617 adet veri bulunmaktadır.

 
<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/img1.png" alt="alt text" width="320" height="280">

%28.7 si Other sınıfının içerisindeyken %71.3'ü Irkçı, Cinsiyetçi, Sexist,Küfür sınıfının içerisindedir.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/img2.png" alt="alt text" width="620" height="280" >

### Veri Ön İşleme

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

* Veri setini temizledikten sonra ön işleme adımlarından olan tokenize,mapping ve encoder dönüşümünü yapılmıştır.

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
### Adım 4 "Name Mapping" işleminden sonra veri setimizin ilk 5 elemanı inceleyelim.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/img0.png" alt="alt text" width="420" height="280">

### Literatür
---

Bert (Bidirectional Encoder Representations from Transformers) modeli, doğal dil işlemede son derece başarılı bir modeldir ve birçok farklı NLP görevinde kullanılabilir.Bert modelini tercih etmeden önce Google Schloar'dan makaleler incelenerek literatür taraması yapılmıştır proje için en uygun model BERT olduğuna karar verilmiştir. Bu model, özellikle kelime anlamının bağlamsal olarak belirlenmesinde ve metinler arasındaki ilişkilerin anlaşılmasında çok başarılıdır.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/literatür.png" alt="alt text" width="520" height="280">



### Modeller
---

Proje gelişimi boyunca, BERTurk'ün *cased* ve *uncased* versiyonları kullanılmıştır. Bu versiyonların üzerinde etkisiz sözcüklerin etkileri ortaya çıkmıştır.


|# | F1-score | Time |
|------------|-------|------|
|BERT-cased-with-stopwords| %89 |29:36 dk|
|BERT-cased-sithout-stopwords| %91|29:15 dk|
|BERT-uncased-sithout-stopwords| %92|28:63 dk|
|BERT-uncased-sith-stopwords| %91| 28:64 dk|

* Projenin canlıya alınmasında **BERT-uncased-with-stopwords** modeli kullanılmıştır.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/confmatrix.png" alt="alt text" width="520" height="280">

### GAT0R SEARCH
---

Flask kullanarak girilen metnin sınıflandırmasını yapan bir uygulama geliştirdik.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/flask-resim.png?raw=true" alt="flask uygulamasi">

### HuggingFace Spaces
---

Modelimizin Gradio servisine aşağıdaki linkden ulaşabilirsiniz.
[Gradio](https://huggingface.co/spaces/thealper2/gat0r-gradio)

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/huggingface-gradio.png?raw=true" alt="gradio">


### Kaynakça
---

[Text Classification with BERT in PyTorch](https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f)

