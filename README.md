MiniTR-GPT

Bu projede PyTorch kullanarak karakter seviyesinde çalışan küçük bir GPT modeli geliştirdim. Amacım hazır Transformer modellerini kullanmak yerine GPT'nin temel çalışma mantığını öğrenmekti.

Projede Neler Var?

Karakter seviyeli tokenizer

Token ve pozisyon embeddingleri

Q, K ve V hesapları

Causal masked multi-head self-attention

Transformer blokları

Cross-entropy loss

AdamW optimizasyonu

Train/validation takibi

Checkpoint kaydetme

Temperature ve top-k ile metin üretimi

Projede nn.Transformer ve nn.MultiheadAttention kullanılmadı.

Model Ayarları

Transformer katmanı: 2

Attention head: 2

Embedding boyutu: 64

Bağlam uzunluğu: 64 karakter

Vocabulary: 65 karakter

Toplam öğrenilebilir parametre: 108.352

İlk Eğitim Sonuçları

Model küçük bir Türkçe demo veri seti üzerinde 300 adım eğitildi.

Eğitim verisi: 9.141 karakter

Validation verisi: 1.016 karakter

Başlangıç validation loss: 4.2046

Son validation loss: 2.6524

Validation perplexity: yaklaşık 14.18

Otomatik testler: 5/5 başarılı

Loss değerinin düşmesi modelin Türkçe karakter geçişlerini, boşlukları ve bazı kelime yapılarını öğrenmeye başladığını gösteriyor.

Kullanılan veri ve eğitim süresi küçük olduğu için model henüz anlamlı ve tutarlı Türkçe metin üretmiyor. Bu sürümün amacı yüksek kaliteli bir dil modeli oluşturmak değil, Transformer mimarisini uçtan uca çalıştırmaktı.

Kurulum

python -m venv venv

Windows:

venv\Scripts\activate

Paketleri yüklemek için:

python -m pip install -r requirements.txt

Çalıştırma

PyCharm üzerinden demo.py dosyasına sağ tıklayıp çalıştırabilirsiniz.

Terminalden:

python demo.py

Testleri çalıştırmak için:

pytest -q

Sonraki Hedefler

Daha büyük ve temiz bir Türkçe veri seti kullanmak

Eğitim süresini artırmak

BPE tokenizer eklemek

Attention haritalarını görselleştirmek

Loss grafiklerini README'ye eklemek

