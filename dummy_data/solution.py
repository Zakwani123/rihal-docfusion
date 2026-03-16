import os
import re
import json
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class DocFusionSolution:
    
    def __init__(self):
        self.model_dir = None
    
    def train(self, train_dir: str, work_dir: str) -> str:
        os.makedirs(work_dir, exist_ok=True)
        config = {'approach': 'rules_based', 'version': '2.0'}
        with open(os.path.join(work_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        return work_dir
    
    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        test_jsonl = os.path.join(data_dir, 'test.jsonl')
        predictions = []
        
        with open(test_jsonl, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                record_id = record['id']
                
                if 'text' in record and record['text']:
                    text = record['text']
                else:
                    img_path = os.path.join(data_dir, 'images', record.get('image', ''))
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        text = pytesseract.image_to_string(img)
                    else:
                        text = ''
                
                vendor = self._extract_vendor(text)
                date = self._extract_date(text)
                total = self._extract_total(text)
                is_forged = self._detect_anomaly(text, vendor, date, total)
                
                predictions.append({
                    'id': record_id,
                    'vendor': vendor,
                    'date': date,
                    'total': total,
                    'is_forged': is_forged
                })
        
        with open(out_path, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
    
    def _extract_date(self, text):
        patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{4}/\d{2}/\d{2}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{2}(?!\d)',
            r'\d{2}-\d{2}-\d{2}(?!\d)',
            r'\d{2}\.\d{2}\.\d{2}(?!\d)',
            r'\d{2}\.\d{2}\.\d{4}',
        ]
        lines = text.strip().split('\n')
        for line in lines:
            if 'date' in line.lower():
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        return match.group()
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None
    
    def _extract_total(self, text):
        lines = text.strip().split('\n')
        skip_words = ['adjustment', 'rounding adj', 'subtotal', 'sub total',
                      'change', 'cash', 'gst', 'tax code', 'tax ']
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if 'adjustment' in line_lower:
                continue
            if 'round' in line_lower and 'total' in line_lower:
                numbers = re.findall(r'\d+\.?\d{0,2}', line)
                valid = [n for n in numbers if '.' in n and float(n) > 0.5]
                if valid:
                    return valid[-1]
                for j in range(i+1, min(i+4, len(lines))):
                    numbers = re.findall(r'\d+\.\d{2}', lines[j])
                    if numbers:
                        return numbers[-1]
        
        total_keywords = ['total (rm)', 'total amount', 'grand total',
                          'total sales', 'total inclusive', 'total:',
                          'total rm', 'nett total', 'net total', 'total amt']
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(skip in line_lower for skip in skip_words):
                continue
            for keyword in total_keywords:
                if keyword in line_lower:
                    numbers = re.findall(r'\d+\.\d{2}', line)
                    if numbers:
                        return numbers[-1]
                    for j in range(i+1, min(i+4, len(lines))):
                        numbers = re.findall(r'\d+\.\d{2}', lines[j])
                        if numbers:
                            return numbers[-1]
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(skip in line_lower for skip in skip_words):
                continue
            if re.match(r'^total[\s:]*', line_lower):
                numbers = re.findall(r'\d+\.\d{2}', line)
                if numbers:
                    return numbers[-1]
                for j in range(i+1, min(i+4, len(lines))):
                    numbers = re.findall(r'\d+\.\d{2}', lines[j])
                    if numbers:
                        return numbers[-1]
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(skip in line_lower for skip in skip_words):
                continue
            if 'total' in line_lower:
                numbers = re.findall(r'\d+\.\d{2}', line)
                if numbers:
                    return numbers[-1]
                for j in range(i+1, min(i+4, len(lines))):
                    numbers = re.findall(r'\d+\.\d{2}', lines[j])
                    if numbers:
                        return numbers[-1]
        
        all_numbers = re.findall(r'\d+\.\d{2}', text)
        if all_numbers:
            return all_numbers[-1]
        return None
    
    def _extract_vendor(self, text):
        lines = text.strip().split('\n')
        skip_indicators = ['tax invoice', 'receipt', 'invoice', 'tel:', 'tel ',
                           'fax:', 'fax ', 'gst', 'date', 'time', 'cashier',
                           'no.', 'lot ', 'jalan', 'taman', 'www.', 'http',
                           'cash', 'member', 'document', 'address']
        business_indicators = ['sdn bhd', 'sdn. bhd', 'sdn.bhd', 'enterprise',
                              'trading', 'restaurant', 'market', 'store', 'shop',
                              'cafe', 'bakery', 'pharmacy', 'holdings', 'bhd',
                              's/b', 'industries', 'corporation', 'company',
                              'supermarket', 'mini market', 'hardware']
        for line in lines[:10]:
            line_clean = line.strip()
            if len(line_clean) < 3:
                continue
            if any(biz in line_clean.lower() for biz in business_indicators):
                return line_clean
        for line in lines[:6]:
            line_clean = line.strip()
            if len(line_clean) < 3:
                continue
            if any(skip in line_clean.lower() for skip in skip_indicators):
                continue
            if re.match(r'^[\d\s\-\+\(\)\.]+$', line_clean):
                continue
            words = line_clean.split()
            if len(words) <= 3 and all(w.isalpha() for w in words) and len(line_clean) < 20:
                continue
            return line_clean
        return None
    
    def _detect_anomaly(self, text, vendor, date, total):
        score = 0
        if vendor is None:
            score += 1
        if date is None:
            score += 1
        if total is None:
            score += 1
        if total:
            try:
                total_val = float(total)
                if total_val > 5000:
                    score += 1
                if total_val <= 0:
                    score += 1
            except:
                pass
        if len(text.strip()) < 50:
            score += 1
        lines = text.strip().split('\n')
        if len(lines) < 5:
            score += 1
        return 1 if score >= 2 else 0

