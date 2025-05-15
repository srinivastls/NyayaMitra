import fitz  # PyMuPDF
import json

def extract_text_by_headings(pdf_path):
    # Open the provided PDF file
    doc = fitz.open(pdf_path)
    headings = {}
    current_heading = None
    
    # Iterate through each page
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if 'lines' in block:  # Ensure block contains lines of text
                for line in block['lines']:
                    spans = line['spans']
                    if spans and 'size' in spans[0] and spans[0]['size'] > 13:  # Example size threshold for headings
                        current_heading = spans[0]['text'].strip()
                        headings[current_heading] = []
                    elif current_heading:
                        headings[current_heading].append(spans[0]['text'].strip())
    
    # Close the document
    doc.close()
    return headings

def save_as_json(data, json_path):
    # Write the extracted data to a JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Path to the PDF and JSON file
pdf_path = "/home/ubuntu/ram/code/data/Greater_Hyderabad_Municipal_Corporation_Act_1955.PDF"
json_path = "output.json"

# Extract text by headings
extracted_data = extract_text_by_headings(pdf_path)

# Save the data as JSON
save_as_json(extracted_data, json_path)

