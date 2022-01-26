# Detect Vaccination

- Detect whether the vaccination symbol is there with yolov5
- Detect the location for the place with pytesseract
- Return whether the phone is showing valid or not

# Setup

```
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) by following instructions [here](https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i)

Within `predict.py`, update the path for tesseract accordingly

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

# Resources

- https://github.com/ultralytics/yolov5