# MiniTR-GPT

Bu projeyi, decoder-only Transformer mimarisinin iç yapısını öğrenmek için hazırladım. Amaç büyük bir dil modeli üretmek değil; GPT'nin temel parçalarını hazır Transformer sınıfları kullanmadan kurmak ve eğitim sürecini uçtan uca görebilmek.

Projede `torch.nn.MultiheadAttention`, `torch.nn.Transformer` veya önceden eğitilmiş GPT ağırlıkları kullanılmıyor. Q, K ve V projeksiyonları, causal mask, attention hesabı, Transformer bloğu, eğitim döngüsü ve metin üretimi proje içinde yazıldı.

## Şu anki durum

- Karakter seviyeli tokenizer çalışıyor.
- Masked multi-head self-attention sıfırdan yazıldı.
- Eğitim ve doğrulama kaybı takip ediliyor.
- Checkpoint kaydetme ve eğitime devam etme var.
- Temperature ve top-k ile metin üretilebiliyor.
- Attention ağırlıkları metin tablosu olarak incelenebiliyor.
- Windows ve PyCharm üzerinde küçük demo akışı test edildi.

## Modelin akışı

```text
metin
  ↓
karakter tokenları
  ↓
token embedding + pozisyon embedding
  ↓
N adet Transformer bloğu
  ├─ layer normalization
  ├─ masked multi-head self-attention
  ├─ residual bağlantı
  ├─ feed-forward ağ
  └─ residual bağlantı
  ↓
son layer normalization
  ↓
her karakter için logit
  ↓
softmax / örnekleme
  ↓
sıradaki karakter
```

Attention hesabının özeti:

```text
softmax((Q × Kᵀ) / √head_size) × V
```

Causal mask nedeniyle bir token yalnızca kendisini ve önceki tokenları görebilir.

## Dosyalar

```text
MiniTR-GPT/
├── data/
│   └── sample_turkish.txt
├── tests/
│   ├── test_model.py
│   └── test_tokenizer.py
├── checkpoints/
├── config.py
├── data_utils.py
├── demo.py
├── generate.py
├── inspect_attention.py
├── model.py
├── prepare_wikipedia.py
├── tokenizer.py
├── train.py
├── requirements.txt
├── DATA_LICENSE.md
├── LICENSE
└── README.md
```

## PyCharm ile kurulum

1. Klasörü PyCharm ile aç.
2. `File > Settings > Project > Python Interpreter` yoluna gir.
3. Proje klasöründe `venv` veya `.venv` adında yeni bir Virtualenv oluştur.
4. PyCharm terminalinde aşağıdaki komutu çalıştır:

```powershell
python -m pip install -r requirements.txt
```

`pip` komutu tek başına bulunamazsa daima `python -m pip` kullanılabilir.

## çalıştırma

PyCharm içinde `demo.py` dosyasına sağ tıklayıp **Run 'demo'** seçilebilir. Bu dosya önce küçük eğitim çalıştırır, ardından oluşan checkpoint ile metin üretir.

Terminalden aynı işlem:

```powershell
python demo.py
```

Windows için eski yöntem de duruyor:

```powershell
.\run_demo.bat
```

Demo verisi küçük olduğu için ilk çıktılar anlamsız olabilir. Bu, modelin çalışmadığı anlamına gelmez. Burada bakılması gereken ilk şey train ve validation loss değerlerinin zamanla düşmesidir.

## Eğitimi elle başlatma

```powershell
python train.py `
  --data data/sample_turkish.txt `
  --out-dir checkpoints/demo `
  --max-iters 300 `
  --eval-interval 50 `
  --eval-iters 10 `
  --batch-size 16 `
  --block-size 64 `
  --n-layer 2 `
  --n-head 2 `
  --n-embd 64
```

Eğitim sonunda şu dosyalar oluşur:

- `best.pt`: en düşük validation loss değerine sahip checkpoint
- `last.pt`: son adımdaki checkpoint
- `tokenizer.json`: karakter sözlüğü
- `training_config.json`: kullanılan ayarlar
- `metrics.jsonl`: loss kayıtları

## Metin üretme

```powershell
python generate.py `
  --checkpoint checkpoints/demo/best.pt `
  --prompt "Bilim" `
  --max-new-tokens 300 `
  --temperature 0.8 `
  --top-k 20
```

Temperature yükseldikçe çıktı çeşitlenir fakat hata ihtimali de artar. Top-k küçüldükçe model daha sınırlı seçenek arasından seçim yapar.


## Daha büyük eğitim örneği

```powershell
python train.py `
  --data data/turkish_wikipedia.txt `
  --out-dir checkpoints/wiki-small `
  --max-iters 3000 `
  --eval-interval 200 `
  --eval-iters 50 `
  --batch-size 32 `
  --block-size 128 `
  --n-layer 4 `
  --n-head 4 `
  --n-embd 128 `
  --learning-rate 3e-4
```

CPU üzerinde bu eğitim uzun sürebilir. GPU varsa `--device cuda` kullanılabilir.

## Eğitime devam etme

```powershell
python train.py `
  --data data/turkish_wikipedia.txt `
  --out-dir checkpoints/wiki-small `
  --resume checkpoints/wiki-small/last.pt `
  --max-iters 5000
```

## Attention inceleme

```powershell
python inspect_attention.py `
  --checkpoint checkpoints/wiki-small/best.pt `
  --text "İstatistik bilimi verileri inceler"
```

Bu komut seçilen attention head'inin son karakter için önceki karakterlere verdiği ağırlıkları gösterir.

## Testler

```powershell
python -m pytest -q
```

Testlerde tokenizer dönüşümü, model boyutları, loss, causal mask ve üretim fonksiyonu kontrol ediliyor.

## Öğrenirken özellikle dikkat ettiğim noktalar

- Token ID bir anlam değeri değil, embedding tablosundaki satır numarasıdır.
- Embedding boyutları tek tek isimlendirilmiş özellikler değildir; bilgi vektöre dağılır.
- Q ile K arasındaki benzerlik attention ağırlığını belirler.
- Model her adımda bir sonraki karakterin koşullu olasılığını öğrenir.
- Cross-entropy, doğru karaktere düşük olasılık verildiğinde büyür.
- Validation loss yükselirken train loss düşüyorsa aşırı öğrenme başlamış olabilir.

## Henüz geliştirilmesi gerekenler

- Daha büyük ve temiz Türkçe veriyle uzun eğitim
- BPE tokenizer deneyi
- Perplexity ve tekrar oranı ölçümü
- Attention haritasını görsel olarak çizme
- Eğitim sonuçlarını karşılaştıran deney tablosu


