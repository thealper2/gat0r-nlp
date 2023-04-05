
### AMAÇ 

Türkçe Doğal Dil İşleme ile özellikle Türkçe metinlerin işlenmesi için gerekli kullanıcı dostu ve yüksek performanslı kütüphanelerin, veri kümelerinin hazırlanmasına katkı sağlamak amaçlanmaktadır. Aşağılayıcı Söylemlerin Doğal Dil İşleme İle Tespiti sağlanacaktır. 
# Adım 1 Veri setini inceleyelim

| # |  Column   |Non-Null| Count|  Dtype |
|---|  ------ |  --------|------ | -----  |
| 0 | id      |12617 | non-null   |object  |
| 1 | text     |12617 |non-null   |object |
|2  | is_offensive|12617 |non-null |int64 |
|3  | target  |12617 |non-null |object|

* NOT eklenecek

#### Adım 2 Veri Setini Temizleden Önce En sık kullanılan keimeleri tespit edilmiştir

* görsel eklenecek

#### Adım 3 veri seti temizlenmiştir

* görsel eklenecek

#### Adım 4 Veri setini temizledikten sonra ön işleme adımlarından olan tokenize,mapping ve encoder dönüşümünü yapılmıştır.

* görsel eklenecek

#### Adım 5 model seçimi yapılmıştır. 
Bert (Bidirectional Encoder Representations from Transformers) modeli, doğal dil işlemede son derece başarılı bir modeldir ve birçok farklı NLP görevinde kullanılabilir.Bert modelini tercih etmeden önce Google Schloar'dan makaleler incelenerek literatür taraması yapılmıştır proje için en uygun model BERT olduğuna karar verilmiştir. Bu model, özellikle kelime anlamının bağlamsal olarak belirlenmesinde ve metinler arasındaki ilişkilerin anlaşılmasında çok başarılıdır.
* Görsel eklenecek



#### Adım 6 Bert modelini seçtikten sonra hangi BERT'in kullanılacağına karar verilmiştir

Bu adımda 4 Bert modeli kullanılmıştır;
##### BERT-Cased-With-Stopword
##### BERT-Cased-Without-Stopword
##### BERT-Uncased-With-Stopword
##### BERT-Uncased-Without-Stopword


|#  |Accuracy| f1-score | Time|
|---------|---------|-------|------|
|Bert-Cased-With-Stopword| | %89 |29:36 dk|
|Bert-Cased-Without-Stopword| | %91|29:15 dk|
|Bert-Uncased-Without-Stopword| | %92|28:63 dk|
|Bert-Uncased-With-Stopword| | %91| 28:64 dk|
