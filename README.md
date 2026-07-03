# 🤖 CodeReviewBot

YZM358 Doğal Dil İşleme dersi kapsamında geliştirilen, kod incelemesi ve onarımı yapan yapay zeka destekli bir sistem.

## 🔍 Nasıl Çalışır?

Kullanıcının girdiği kod üç aşamalı bir pipeline'dan geçer:

```
Kod Girişi
    ↓
AST Parse → Graf Kodlama (Node2Vec)
    ↓
FAISS ile Benzer Örnekler Getirilir (RAG)
    ↓
Fine-tuned CodeT5+ → İnceleme + Onarım Önerisi
```

1. **AST Graf Kodlama** — `tree-sitter` ile kod soyut sözdizim ağacına (AST) dönüştürülür, `Node2Vec` ile vektör temsiline çevrilir.
2. **RAG** — `FAISS` indeksi üzerinden benzer kod/inceleme çiftleri getirilip modele bağlam olarak sunulur.
3. **CodeT5+ ile Üretim** — Fine-tune edilmiş model hem kod incelemesi hem de kod onarımı (code repair) üretir.
4. **Gradio Arayüzü** — Kullanıcı dostu arayüzde Python ve Java desteklenir.

## 📁 Klasör Yapısı

```
CodeReviewBot/
├── notebooks/
│   ├── Uye1_AST_GraphBERT.ipynb      # AST parse + graf kodlama + eğitim
│   ├── Uye2_CodeRepair.ipynb          # Code repair modeli eğitimi
│   └── Uye3_Arayuz_v2.ipynb          # RAG pipeline + Gradio arayüzü
├── utils/                             # Yardımcı script'ler
├── models/                            # Fine-tuned model checkpoint'leri (Drive'da)
└── model_ciktisi/                     # FAISS indeksi, final model (Drive'da)
```

> 📦 **Model ağırlıkları ve dataset boyut nedeniyle repoya dahil edilmemiştir.**  
> Erişim için: [Google Drive Klasörü](https://drive.google.com/drive/folders/1AzI7GtKcRnifC0R_9nu7UhVCyqrQIiXY)

## 🛠️ Kullanılan Teknolojiler

- `tree-sitter` — Python ve Java AST parsing
- `node2vec` — Graf vektör temsili
- `transformers` — CodeT5+ (Salesforce/codet5p-220m)
- `faiss-cpu` — Vektör benzerlik arama
- `gradio` — İnteraktif web arayüzü

## 🚀 Kurulum

```bash
pip install node2vec==0.4.6
pip install numpy==1.26.4
pip install tree-sitter==0.21.3 tree-sitter-python==0.21.0 tree-sitter-java==0.21.0
pip install transformers==4.41.0 sentencepiece networkx gradio faiss-cpu
```

## 📌 Kullanım

Notebook'ları sırasıyla çalıştır:
1. `Uye1_AST_GraphBERT.ipynb` → Graf kodlama ve review modeli eğitimi
2. `Uye2_CodeRepair.ipynb` → Code repair modeli eğitimi  
3. `Uye3_Arayuz_v2.ipynb` → Pipeline'ı birleştir ve Gradio arayüzünü çalıştır

---

*Samsun Üniversitesi — YZM358 Doğal Dil İşleme Projesi, 2025-2026*
