import sys

# Windows: ถ้า output ถูก redirect/pipe จะใช้ cp1252 แล้ว print ภาษาไทยไม่ได้ (UnicodeEncodeError)
# ไฟล์นี้ถูก import ทุกครั้งที่รัน python -m src.xxx จึงแก้ครั้งเดียวที่นี่พอ
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
