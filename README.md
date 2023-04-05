# :crocodile: GAT0R - Türkçe Doğal Dil İşleme Ekibi
---

<img  src="https://github.com/thealper2/gat0r-nlp/blob/main/flask/logo.jpg" alt="alt text" width="320" height="280">

*Bu çalışma, Teknofest 2023 Türkçe Doğal Dil İşleme yarışması kategorisinde "Aşağılayıcı Dil Tespiti" için geliştirilmiştir.*
### :crocodile: Amaç
---

Türkçe Doğal Dil İşleme ile özellikle Türkçe metinlerin işlenmesi için gerekli kullanıcı dostu ve yüksek performanslı kütüphanelerin, veri kümelerinin hazırlanmasına katkı sağlamak amaçlanmaktadır. Aşağılayıcı söylemlerin doğal dil işleme ile tespiti sağlanacaktır. Tespit edilen söylemler; SEXIST, INSULT, PROFANITY, RACIST ve OTHER olarak sınıflandırılacaktır. 


### :crocodile: Veri setini inceleyelim.
---

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
* "Name Mapping" işleminden sonra veri setimizin ilk 5 elemanı inceleyelim.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/img0.png" alt="alt text" width="420" height="280">

### :crocodile: Modeller
---

Proje gelişimi boyunca, BERTurk'ün *cased* ve *uncased* versiyonları kullanılmıştır. Bu versiyonların üzerinde etkisiz sözcüklerin etkileri ortaya çıkmıştır.


|# | F1-score | Time |
|------------|-------|------|
|BERT-cased-with-stopwords| %89 |29:36 dk|
|BERT-cased-without-stopwords| %91|29:15 dk|
|BERT-uncased-without-stopwords| %92|28:63 dk|
|BERT-uncased-with-stopwords| %91| 28:64 dk|

* Projenin canlıya alınmasında **BERT-uncased-with-stopwords** modeli kullanılmıştır. Modellerin kodlarına gitmek için aşağıdaki linklere tıklayın.

[BERT-cased-with-stopwords](https://github.com/thealper2/gat0r-nlp/blob/main/models/BERT_cased_with_stopwords.ipynb)
[BERT-cased-without-stopwords](https://github.com/thealper2/gat0r-nlp/blob/main/models/BERT_cased_without_stopwords.ipynb)
[BERT-uncased-without-stopwords](https://github.com/thealper2/gat0r-nlp/blob/main/models/BERT_uncased_without_stopwords.ipynb)
[BERT-uncased-with-stopwords](https://github.com/thealper2/gat0r-nlp/blob/main/models/BERT_uncased_with_stopwords.ipynb)


<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/confmatrix.png" alt="alt text" width="520" height="280">

### :crocodile: GAT0R SEARCH
---

Flask kullanarak girilen metnin sınıflandırmasını yapan bir uygulama geliştirdik.

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/flask-resim.png?raw=true" alt="flask uygulamasi">

### :crocodile: HuggingFace Spaces
---

Modelimizin Gradio servisine aşağıdaki linkden ulaşabilirsiniz.
[Gradio](https://huggingface.co/spaces/thealper2/gat0r-gradio)
[Kaynak Kodu](https://github.com/thealper2/gat0r-nlp/blob/main/NLPEvaluation_GAT0R.py)

<img src="https://github.com/thealper2/gat0r-nlp/blob/main/images/huggingface-gradio.png?raw=true" alt="gradio">


### :crocodile: Kaynaklar
---

[Text Classification with BERT in PyTorch](https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f)

