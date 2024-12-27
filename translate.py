import pandas as pd

# Đọc file để xem trước dữ liệu
file_source = "/content/drive/MyDrive/MIMIC-Data/physionet.org/files/mimic3-carevue/1.4/NOTEEVENTS.csv.gz"
df_source = pd.read_csv(file_source, compression="gzip")  # Giải nén và đọc file

# Hiển thị thông tin tổng quan
print(df_source.info())
print(df_source.head())

df_sample = df_source.sample(frac=0.01, random_state=42)  # Lấy 5% dữ liệu ngẫu nhiên
print(f"Số lượng dòng sau khi lấy mẫu: {len(df_sample)}")

from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

# Load mô hình dịch
model_name = "Helsinki-NLP/opus-mt-en-vi"  # Dùng mô hình EN-VI
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Hàm dịch theo batch
def translate_batch(texts, batch_size=32):
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**tokens)
        batch_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        translations.extend(batch_translations)
    return translations

# Dịch cột 'TEXT'
df_sample["Text Dịch"] = translate_batch(df_sample["text"].dropna().tolist())

# Xuất file song ngữ
output_path = "song_ngu_en_vi_NOTEEVENTS.csv"
df_sample[["TEXT", "Text Dịch"]].to_csv(output_path, index=False, encoding="utf-8")