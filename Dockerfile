# Gunakan image Python resmi sebagai base image yang ringan
# Menggunakan versi slim dianjurkan untuk ukuran image yang lebih kecil
FROM python:3.11.4-slim

# Tetapkan direktori kerja di dalam kontainer
WORKDIR /app

# 1. Salin requirements.txt terlebih dahulu
# Ini memanfaatkan Docker's layer caching. Jika requirements.txt tidak berubah, 
# langkah instalasi tidak perlu dijalankan lagi.
COPY requirements.txt .

# 2. Instal semua dependensi Python
# '--no-cache-dir' mengurangi ukuran image
RUN pip install --no-cache-dir -r requirements.txt

# 3. Salin sisa kode aplikasi/proyek MLOps ke dalam kontainer
# Pastikan Anda menggunakan file .dockerignore untuk mengecualikan file besar 
# atau tidak perlu (misalnya: .git, __pycache__, data, model yang sudah dilatih)
COPY . .

# 4. (Opsional) Paparkan port jika aplikasi Anda adalah API/layanan inferensi
# Ganti 8000 dengan port yang digunakan oleh aplikasi Anda (misalnya Flask/FastAPI)
# EXPOSE 8000 

# 5. Tentukan perintah yang akan dijalankan saat kontainer dimulai
# Ganti 'python main.py' dengan perintah yang sesuai untuk menjalankan 
# alur kerja MLOps atau server API Anda.

# Contoh untuk menjalankan skrip utama:
# CMD ["python", "train.py"] 

# Contoh untuk menjalankan server FastAPI/Uvicorn (jika proyek Anda adalah deployment API):
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]