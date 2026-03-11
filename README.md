Demo Object Detection berbasis YOLO menggunakan FastAPI.

Upload single atau multiple images.

Deteksi objek otomatis menggunakan model YOLO (best.pt).

Preview gambar dengan bounding box dan confidence setiap objek.

Zoom gambar via overlay saat diklik.

Pilih objek tertentu untuk disimpan, bisa save beberapa sekaligus dalam zip.

Hapus gambar sebelum deteksi jika tidak ingin diproses.

Struktur project:

app/ – FastAPI backend

models/ – Model YOLO (best.pt)

static/ – Frontend HTML/JS demo

logs/ – Log hasil deteksi

requirements.txt – Dependensi Python

Instruksi singkat menjalankan project:

Clone repo: git clone <repo-url>

Masuk folder: cd FastApi-Object-Detection

Buat virtual environment: python -m venv venv

Aktifkan venv: venv\Scripts\activate (Windows) / source venv/bin/activate (Linux/Mac)

Install requirements: pip install -r requirements.txt

Jalankan API: uvicorn app.main:app --reload

Buka API docs: http://127.0.0.1:8000/docs

Frontend demo: buka static/frontend.html di browser"
