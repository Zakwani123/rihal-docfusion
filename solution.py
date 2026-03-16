import os
import re
import json
import pickle
import pytesseract
from PIL import Image
import csv
import numpy as np

import platform
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class DocFusionSolution:

    def __init__(self):
        self.model_dir = None
        self.anomaly_model = None

    def train(self, train_dir: str, work_dir: str) -> str:
        os.makedirs(work_dir, exist_ok=True)

        train_jsonl = os.path.join(train_dir, 'train.jsonl')

        known_vendors = set()
        vendor_totals = {}

        with open(train_jsonl, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                fields = record.get('fields', {})

                vendor = fields.get('vendor')
                total = fields.get('total')

                if vendor:
                    known_vendors.add(vendor)

                if vendor and total:
                    if vendor not in vendor_totals:
                        vendor_totals[vendor] = []
                    vendor_totals[vendor].append(float(re.sub(r'[^\d.]', '', str(total))))

        vendor_ranges = {}
        for vendor, totals in vendor_totals.items():
            vendor_ranges[vendor] = {
                'min': min(totals),
                'max': max(totals)
            }

        all_totals = []
        for totals in vendor_totals.values():
            all_totals.extend(totals)

        if all_totals:
            overall_stats = {
                'min': min(all_totals),
                'max': max(all_totals),
                'mean': sum(all_totals) / len(all_totals)
            }
        else:
            overall_stats = {'min': 0, 'max': 9999, 'mean': 200}

        config = {
            'known_vendors': list(known_vendors),
            'vendor_ranges': vendor_ranges,
            'overall_stats': overall_stats
        }
        with open(os.path.join(work_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        self._train_anomaly_model(train_dir, work_dir)

        self.model_dir = work_dir
        return work_dir

    def _train_anomaly_model(self, train_dir, work_dir):
        """Train anomaly model on Find-It-Again data if available."""
        from sklearn.ensemble import RandomForestClassifier

        # check a few likely paths for the findit2 data
        possible_paths = [
            os.path.join(train_dir, 'findit2', 'train.txt'),
            os.path.join(train_dir, '..', 'findit2', 'train.txt'),
            os.path.join(train_dir, '..', '..', 'Datasets', 'findit2', 'train.txt'),
            r'C:\Users\modyx\OneDrive\Desktop\Rihal Project\Datasets\findit2\train.txt',
        ]

        findit_path = None
        findit_dir = None
        for p in possible_paths:
            if os.path.exists(p):
                findit_path = p
                findit_dir = os.path.join(os.path.dirname(p), 'train')
                break

        if findit_path is None or not os.path.isdir(findit_dir):
            print("[INFO] Find-It-Again dataset not found. Skipping anomaly model training.")
            return

        samples = []
        with open(findit_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 4:
                    continue
                filename = row[0].strip()
                try:
                    forged = int(row[3].strip())
                except ValueError:
                    continue
                samples.append((filename, forged))

        features = []
        labels = []
        for filename, forged in samples:
            txt_file = os.path.join(findit_dir, filename.replace('.png', '.txt'))
            if not os.path.exists(txt_file):
                continue

            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except:
                continue

            feat = self._extract_anomaly_features(text)
            features.append(feat)
            labels.append(forged)

        if len(features) < 10:
            print(f"[INFO] Only {len(features)} samples found. Skipping anomaly model training.")
            return

        X = np.array(features)
        y = np.array(labels)

        # balanced weighting since only ~16% are forged
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X, y)

        model_path = os.path.join(work_dir, 'anomaly_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        self.anomaly_model = model
        print(f"[INFO] Anomaly model trained on {len(features)} samples. "
              f"Forged: {sum(labels)}, Genuine: {len(labels) - sum(labels)}")

    def _extract_anomaly_features(self, text):
        """10 numerical features from the receipt text."""
        lines = text.strip().split('\n')
        num_lines = len(lines)
        text_length = len(text)

        vendor = self._extract_vendor(text)
        date = self._extract_date(text)
        total = self._extract_total(text)

        has_vendor = 1 if vendor else 0
        has_date = 1 if date else 0
        has_total = 1 if total else 0

        total_value = 0.0
        if total:
            try:
                total_value = float(total)
            except:
                total_value = 0.0

        if text_length > 0:
            digit_ratio = sum(1 for c in text if c.isdigit()) / text_length
            upper_ratio = sum(1 for c in text if c.isupper()) / text_length
            special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / text_length
        else:
            digit_ratio = 0
            upper_ratio = 0
            special_ratio = 0

        avg_line_length = text_length / max(num_lines, 1)

        return [
            text_length,
            num_lines,
            has_vendor,
            has_date,
            has_total,
            total_value,
            digit_ratio,
            upper_ratio,
            special_ratio,
            avg_line_length
        ]

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        self.model_dir = model_dir

        anomaly_model_path = os.path.join(model_dir, 'anomaly_model.pkl')
        if os.path.exists(anomaly_model_path):
            with open(anomaly_model_path, 'rb') as f:
                self.anomaly_model = pickle.load(f)

        test_jsonl = os.path.join(data_dir, 'test.jsonl')
        predictions = []

        with open(test_jsonl, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                record_id = record['id']

                if 'text' in record and record['text']:
                    text = record['text']
                else:
                    img_path = os.path.join(data_dir, record.get('image_path', ''))
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
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{2}/\d{2}/\d{2}(?!\d)',
            r'\d{2}-\d{2}-\d{2}(?!\d)',
            r'\d{2}\.\d{2}\.\d{2}(?!\d)',
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}',
            r'\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{2,4}',
        ]
        lines = text.strip().split('\n')
        for line in lines:
            if 'date' in line.lower():
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group()
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()
        return None

    def _extract_total(self, text):
        # fix OCR spacing errors like "39. 80" -> "39.80"
        text = re.sub(r'(\d+)\.\s+(\d{2})\b', r'\1.\2', text)

        lines = text.strip().split('\n')

        skip_words = ['adjustment', 'rounding adj', 'subtotal', 'sub total',
                      'change', 'cash', 'tax code', 'tax ',
                      'rounding', 'discount', 'sub-total', 'tot qty',
                      'rounding (', 'excluded', 'salesperson', 'receipt#',
                      'totalexcl', 'total excl', 'total qty', 'total item',
                      'totalitem', 'totalamt', 'total amt', 'totalgst',
                      'total gst']

        def clean_number(s):
            # fix common OCR character swaps in numbers
            s = re.sub(r'(?<=\d)[iIl](?=\d)', '1', s)
            s = re.sub(r'(?<=\d)[uU](?=\d)', '0', s)
            s = re.sub(r'(?<=\d)[oO](?=\d)', '0', s)
            s = re.sub(r'(\d),\s*(\d{2})$', r'\1.\2', s.strip())
            match = re.search(r'\d+\.\d{2}', s)
            return match.group() if match else None

        def find_number_in_line_or_next(lines, i):
            numbers = re.findall(r'\d+[.,]\s*\d{2}', lines[i])
            for n in reversed(numbers):
                cleaned = clean_number(n)
                if cleaned:
                    return cleaned
            for j in range(i+1, min(i+5, len(lines))):
                next_lower = lines[j].lower().strip()
                if any(skip in next_lower for skip in skip_words):
                    continue
                numbers = re.findall(r'\d+[.,]\s*\d{2}', lines[j])
                for n in reversed(numbers):
                    cleaned = clean_number(n)
                    if cleaned:
                        return cleaned
            return None

        def find_first_number_same_line(line):
            numbers = re.findall(r'\d+[.,]\s*\d{2}', line)
            for n in numbers:
                cleaned = clean_number(n)
                if cleaned:
                    return cleaned
            return None

        def has_total(s):
            return bool(re.search(r't[o0a]ta[l!i1]', s))

        def should_skip(line_lower):
            return any(skip in line_lower for skip in skip_words)

        # priority 1: rounded/nett total
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if should_skip(line_lower):
                continue
            if ('round' in line_lower and has_total(line_lower)) or 'nett total' in line_lower or 'net total' in line_lower:
                result = find_number_in_line_or_next(lines, i)
                if result:
                    return result

        # priority 2: inclusive total
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if should_skip(line_lower):
                continue
            if has_total(line_lower) and 'incl' in line_lower:
                result = find_number_in_line_or_next(lines, i)
                if result:
                    return result

        # priority 3: specific total keywords
        total_keywords = ['total (rm)', 'grand total', 'total amount',
                          'total sales', 'total inclusive', 'total:',
                          'total rm', 'total aht',
                          'nett total', 'net total', 'rounded total',
                          'round. :d total', 'round total',
                          'torr:', 'tor:']
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if should_skip(line_lower):
                continue
            for keyword in total_keywords:
                if keyword in line_lower:
                    result = find_number_in_line_or_next(lines, i)
                    if result:
                        return result

        # priority 3.5: OCR-garbled "total" + "sales"
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if should_skip(line_lower):
                continue
            if has_total(line_lower) and 'sales' in line_lower:
                result = find_number_in_line_or_next(lines, i)
                if result:
                    return result

        # priority 4: line starts with "total"
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if should_skip(line_lower):
                continue
            if re.match(r'^t[o0a]ta[l!i1][\s:]*', line_lower):
                result = find_first_number_same_line(line)
                if result:
                    return result
                result = find_number_in_line_or_next(lines, i)
                if result:
                    return result

        # priority 4.5: OCR-corrupted numbers on a total line
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if should_skip(line_lower):
                continue
            if has_total(line_lower):
                loose = re.sub(r'[iIl]', '1', line)
                loose = re.sub(r'[uU]', '0', loose)
                loose = re.sub(r'[oO](?=\d)', '0', loose)
                numbers = re.findall(r'\d+\.\d{2}', loose)
                if numbers:
                    return numbers[-1]

        # priority 5: any line with "total", grab first number
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if should_skip(line_lower):
                continue
            if has_total(line_lower):
                result = find_first_number_same_line(line)
                if result:
                    return result

        # priority 6: "RM" followed by a number near the bottom
        for line in reversed(lines[-15:]):
            match = re.search(r'RM\s*(\d+\.\d{2})', line)
            if match:
                return match.group(1)

        return None

    def _extract_vendor(self, text):
        if self.model_dir:
            config_path = os.path.join(self.model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            known_vendors = config.get('known_vendors', [])
            text_lower = text.lower()
            for vendor in known_vendors:
                # normalize before comparing — strip dots, extra spaces
                vendor_clean = vendor.lower().strip().rstrip('.')
                vendor_clean = re.sub(r'\s+', ' ', vendor_clean)
                if vendor_clean in text_lower:
                    return vendor
                # also try with SDN BHD variations
                for variation in [vendor_clean.replace('sdn bhd', 'son bhd'),
                                  vendor_clean.replace('sdn bhd', 's/b'),
                                  vendor_clean.replace('s/b', 'sdn bhd')]:
                    if variation in text_lower:
                        return vendor

        lines = text.strip().split('\n')

        skip_lines = [
            'invoice', 'receipt', 'official receipt', 'tax invoice',
            'cash bill', 'cash sale', 'delivery order', 'quotation',
            'tel', 'fax', 'phone', 'no.', 'jalan', 'lot ',
            'kawasan', 'taman', 'bandar', 'perindustrian',
            'email', 'website', 'www', 'http',
            'thank you', 'terima kasih', 'welcome',
            'date', 'time', 'cashier', 'customer',
            'shopping hours', 'cash', 'bistro & cafe', 'brewery',
            'mon-', 'tue-', 'wed-', 'open daily', 'operating hours',
            'simplified tax', 'gst id', 'gst reg'
        ]

        business_keywords = [
            'sdn bhd', 'sdn. bhd', 'bhd', 'enterprise', 'trading', 'shop',
            'mart', 'store', 'restaurant', 'cafe', 'hotel', 'supermarket',
            'market', 'food', 'bakery', 'pharmacy', 'clinic', 'hardware',
            'auto', 'service', 'centre', 'center', 'co.', 'corp', 'inc',
            'holdings', 'industries', 'retail', 'mini market', 'hypermarket',
            'gallery', 'motor', 'petro', 'asia', 'perniagaan', 'restoran',
            'bistro', 'gift', 'abc', 'speed mart', 'stationery', 's/b'
        ]

        person_indicators = ['mr', 'mrs', 'ms', 'dr', 'sir', 'madam']

        # join lines that got split mid-word by OCR
        joined_lines = []
        i = 0
        while i < min(len(lines), 15):
            current = lines[i].strip()
            if i + 1 < len(lines) and re.search(r'\s[A-Z]$', current):
                next_line = lines[i + 1].strip()
                if next_line and re.match(r'^[A-Za-z]', next_line):
                    current = current + next_line
                    i += 2
                    joined_lines.append(current)
                    continue
            joined_lines.append(current)
            i += 1

        candidates = []

        for idx, line_clean in enumerate(joined_lines[:10]):
            if len(line_clean) < 3:
                continue
            line_lower = line_clean.lower()

            if any(skip in line_lower for skip in skip_lines):
                continue
            if any(line_lower.startswith(p) for p in person_indicators):
                continue
            if re.search(r'\d{5,}', line_clean):
                continue
            if re.match(r'^[\d\s\-\+\(\)\.\,:]+$', line_clean):
                continue

            score = 0

            if any(kw in line_lower for kw in business_keywords):
                score += 10

            alpha_chars = [c for c in line_clean if c.isalpha()]
            if alpha_chars:
                upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                if upper_ratio > 0.6:
                    score += 5

            if idx == 0:
                score += 3
            elif idx <= 2:
                score += 1

            if len(line_clean) > 15:
                score += 1

            candidates.append((score, line_clean))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]

        # if the best candidate is very short, try combining with the next line
        if len(best.split()) <= 2 and len(best) < 15:
            best_idx = None
            for idx, line_clean in enumerate(joined_lines[:10]):
                if line_clean.strip() == best:
                    best_idx = idx
                    break
            if best_idx is not None and best_idx + 1 < len(joined_lines):
                next_line = joined_lines[best_idx + 1].strip()
                next_lower = next_line.lower()
                if not any(skip in next_lower for skip in skip_lines) and len(next_line) > 2:
                    combined = best + ' ' + next_line
                    if any(kw in combined.lower() for kw in business_keywords):
                        best = combined
        # clean up common trailing noise
        best = best.strip().rstrip('.')
        best = re.sub(r'\s*\([\d\w\-\.:]+\)\s*$', '', best)  
        best = re.sub(r'\s+\d[\d\-\.,:\s]+$', '', best)  
        best = re.sub(r'\s+$', '', best)

        return best

    def _detect_anomaly(self, text, vendor, date, total):
        # try the trained model first
        if self.anomaly_model is not None:
            try:
                feat = self._extract_anomaly_features(text)
                prediction = self.anomaly_model.predict([feat])[0]
                return int(prediction)
            except:
                pass

        # try loading from disk
        if self.model_dir:
            model_path = os.path.join(self.model_dir, 'anomaly_model.pkl')
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.anomaly_model = pickle.load(f)
                    feat = self._extract_anomaly_features(text)
                    prediction = self.anomaly_model.predict([feat])[0]
                    return int(prediction)
                except:
                    pass

        # rule-based fallback
        score = 0

        overall_stats = {'min': 0, 'max': 9999, 'mean': 200}
        if self.model_dir:
            config_path = os.path.join(self.model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                overall_stats = config.get('overall_stats', overall_stats)

        if vendor is None:
            score += 2
        if date is None:
            score += 1
        if total is None:
            score += 2

        if total:
            try:
                total_val = float(total)
                o_min = overall_stats.get('min', 0)
                o_max = overall_stats.get('max', 9999)
                if total_val < o_min * 0.7 or total_val > o_max * 1.3:
                    score += 2
            except:
                pass

        if date:
            if not re.match(r'\d{4}-\d{2}-\d{2}', date):
                score += 1

        if len(text.strip()) < 50:
            score += 1

        return 1 if score >= 3 else 0