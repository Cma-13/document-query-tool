import fitz  # PyMuPDF - converts PDF pages to images
import pytesseract
from PIL import Image
import io

def ocr_pdf(pdf_path, lang="nep"):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        # Render the page as an image (higher zoom = better OCR accuracy)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        # Run OCR on the image
        text = pytesseract.image_to_string(image, lang=lang)
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"

    return full_text

if __name__ == "__main__":
    result = ocr_pdf("sample1.pdf", lang="eng+nep")
    
    with open("cleaned_output.txt", "w", encoding="utf-8") as f:
        f.write(result)
    
    print("OCR complete. Output saved to cleaned_output.txt")
    print(f"Total characters extracted: {len(result)}")