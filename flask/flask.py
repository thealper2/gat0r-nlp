# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import torch

# Gerekli kütüphaneler indirilmiştir.
!pip install transformers
!pip install flask-ngrok
!pip install pyngrok

# Ngrok servisi bağlantısı için token yüklenilmiştir.
!ngrok authtoken ''

from transformers import BertTokenizer, BertModel

class BertClassifier(torch.nn.Module):
  def __init__(self, dropout=0.5):
    super(BertClassifier, self).__init__()
    
    self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
    self.dropout = torch.nn.Dropout(dropout)
    # Kullandığımız önceden eğilmiş model "base" sınıfına ait bir BERT modelidir. Yani;
    # 12 layers of Transformer encoder, 12 attention heads, 768 hidden size, 110M parameters.
    # 768, BERT-base modelindeki hidden size'yi, 5 ise veri setimizdeki toplam kategori sayısını temsil ediyor.
    self.linear = torch.nn.Linear(768, 5)
    self.relu = torch.nn.ReLU()

  def forward(self, input_id, mask):
    # _ değişkeni dizideki tüm belirteçlerin gömme vektörlerini içerir.
    # pooled_output değişkeni [CLS] belirtecinin gömme vektörünü içerir.
    # Metin sınıflandırma için polled_output değişkenini girdi olarak kullanmak yeterlidir.

    # Attention mask, bir belirtecin gercek bir kelimemi yoksa dolgu mu olduğunu tanımlar.
    # Eğer gerçek bir kelime ise attention_mask=1, eğer dolgu ise attention_mask=0 olacaktır.
    # return_dict, değeri "True ise" bir BERT modeli tahmin, eğitim veya değerlendirme sırasında ortaya çıkan
    # loss, logits, hidden_states ve attentions dan oluşan bir tuple oluşturacaktır.
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    final_layer = self.relu(linear_output)

    return final_layer

model = BertClassifier()
# Önceden kaydedilen model yüklenmiştir.
model.load_state_dict(torch.load('/content/drive/MyDrive/flask/bert_uncased-with-stopwords.pt', map_location=torch.device('cpu')))

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")

def predict(model, sentence):
  device = torch.device("cuda")
  model = model.cuda()
  # Prediction işlemi sırasında model ağırlıklarını değiştirmeyeceğimiz modelin gradyanlara ihtiyacı yoktur
  # "no_grad" fonksiyonu ile gradyan hesaplarını devre dışı bırakıyoruz.
  with torch.no_grad():
    # text = Modeli eğitmek için kullanılacak veri setindeki "clean_text" sütunundaki her bir satır.
    # padding = Her bir diziyi belirttiğimiz maksimum uzunluga kadar doldurmak için.
    # max_length = Her bir dizinin maksimum uzunluğu
    # truncation = Eğer değeri "True" ise dizimiz maksimum uzunluğu aşar ise onu keser.
    # return_tensors = Döndürelecek tensörlerin türü. Pytorch kullandığımız için "pt" yazıyoruz. Tensorflow kullansaydık "tf" yazmamız gerekirdi.
    input_id = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
    
    # Attention mask, bir belirtecin gercek bir kelimemi yoksa dolgu mu olduğunu tanımlar.
    # Eğer gerçek bir kelime ise attention_mask=1, eğer dolgu ise attention_mask=0 olacaktır.
    mask = input_id['attention_mask'].to(device)
    
    # squeeze() fonksiyonu ile "input_ids" özelliğindeki tensörlerin boyutu 1 olan boyutları
    # kaldırarak, tensörün boyutunu azaltıyoruz.
    input_id = input_id['input_ids'].squeeze(1).to(device)
    
    # Modelin eğitim verileri üzerindeki tahminlerinin sonuçları saklanır.
    output = model(input_id, mask)
    result = output.argmax(dim=1).item()

    categories =  {
        0: 'INSULT',
        1: 'OTHER',
        2: 'PROFANITY',
        3: 'RACIST',
        4: 'SEXIST'
    }
    

    # Kategorik sınıfı döndür.
    return categories.get(output.argmax(dim=1).item())

# Flask için gerekli dosya dizini yapısı oluşturulmuştur.

!mkdir templates
!cp "/content/drive/MyDrive/flask/after.html" templates/after.html
!cp "/content/drive/MyDrive/flask/home.html" templates/home.html
!mkdir static
!mkdir static/stylesheets
!cp "/content/drive/MyDrive/flask/style.css" static/stylesheets/
!cp "/content/drive/MyDrive/flask/logo.jpg" static/logo.jpg
!cp "/content/drive/MyDrive/flask/banner.png" static/banner.png

from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)   
  

@app.route("/")
def main():
  return render_template('home.html')

@app.route("/predict", methods=['POST'])
def home():
  # Kullanıcı girdisi bir değişkene atanmıştır.
  data = request.form['data']
  # Girilen cümle, tahmin fonksiyonuna gönderilmiştir.
  pred = predict(model, data)
  return render_template('after.html', data=pred)

app.run()

