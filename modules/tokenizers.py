"""
Tokenizer for CT2Rep on RadGenome-ChestCT.

ACTUAL CSV format: ['Volumename', 'Anatomy', 'Sentence']
- We collect all 'Sentence' values to build vocabulary
"""

import os
import re
import pickle
from collections import Counter
from pathlib import Path
import pandas as pd

# Paths (matching RadGenome_download.py)
DATA_DIR = "/umbc/rs/pi_oates/users/dta1/Data/Medical_Report_Generation/RadGenome_dataset"
CACHE_DIR = "/umbc/rs/pi_oates/users/dta1/all_cache"


class Tokenizer:
    """
    Tokenizer for radiology reports.
    """
    
    def __init__(self, args):
        """
        Initialize tokenizer.
        """
        self.data_dir = Path(getattr(args, 'data_dir', DATA_DIR))
        self.threshold = getattr(args, 'threshold', 3)
        self.max_seq_length = getattr(args, 'max_seq_length', 300)
        
        # Special tokens
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        
        # Mappings
        self.token2idx = {}
        self.idx2token = {}
        
        # Build vocabulary
        self._build_vocabulary()
        
        # Token IDs
        self.pad_token_id = self.token2idx[self.pad_token]
        self.sos_token_id = self.token2idx[self.sos_token]
        self.eos_token_id = self.token2idx[self.eos_token]
        self.unk_token_id = self.token2idx[self.unk_token]
    
    def _collect_reports(self):
        """
        Collect all sentences from RadGenome-ChestCT dataset.
        CSV format: [Volumename, Anatomy, Sentence]
        """
        sentences = []
        
        # Search paths for CSV files
        dataset_dir = self.data_dir / 'dataset'
        if not dataset_dir.exists():
            dataset_dir = self.data_dir
        
        search_dirs = [
            dataset_dir / 'radgenome_files',
            dataset_dir,
            self.data_dir / 'radgenome_files',
            self.data_dir,
        ]
        
        print(f"  Searching for reports in: {self.data_dir}")
        
        # Find CSV files containing reports
        csv_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                # Look for region_report CSVs specifically
                csv_files.extend(list(search_dir.glob('*region_report*.csv')))
        
        # Remove duplicates
        csv_files = list(set(csv_files))
        print(f"  Found {len(csv_files)} report CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"    Loading: {csv_file.name} ({len(df)} rows, columns: {list(df.columns)})")
                
                # The column is 'Sentence' based on the log output
                sentence_col = None
                for col in df.columns:
                    if col.lower() in ['sentence', 'text', 'report']:
                        sentence_col = col
                        break
                
                if sentence_col:
                    for text in df[sentence_col].dropna():
                        if isinstance(text, str) and len(text) > 10:
                            sentences.append(text)
                    print(f"      Collected {len(df[sentence_col].dropna())} sentences from '{sentence_col}' column")
                else:
                    print(f"      Warning: No sentence column found in {csv_file.name}")
                
            except Exception as e:
                print(f"    Error reading {csv_file}: {e}")
        
        print(f"  Total collected: {len(sentences)} sentences")
        
        # If no sentences found, use default medical vocabulary
        if not sentences:
            print("  Using default medical vocabulary")
            sentences = self._get_default_reports()
        
        return sentences
    
    def _get_default_reports(self):
        """Default medical reports for vocabulary building."""
        return [
            "The lungs are clear bilaterally with no focal consolidation, mass, or nodule. "
            "The heart size is within normal limits. The mediastinum is unremarkable. "
            "No pleural effusion or pneumothorax. The visualized osseous structures are intact.",
            
            "Comparison is made to prior CT examination. The previously noted pulmonary nodule "
            "has decreased in size. No new pulmonary nodules identified. Airways are patent.",
            
            "Mild emphysematous changes predominantly affecting the upper lobes. Scattered areas "
            "of ground-glass opacity in the lower lobes. Main pulmonary artery mildly dilated.",
            
            "No evidence of pulmonary embolism. Thoracic aorta normal caliber without dissection. "
            "Heart and pericardium appear normal. No significant lymphadenopathy.",
            
            "Findings consistent with interstitial lung disease with usual interstitial pneumonia "
            "pattern. Honeycombing present at lung bases. Traction bronchiectasis noted.",
            
            "Post-surgical changes in right hemithorax from prior lobectomy. Remaining lung "
            "parenchyma clear. No evidence of recurrent disease. Surgical clips stable.",
            
            "Multiple bilateral pulmonary nodules identified. Largest measuring 12mm in left "
            "upper lobe. Given malignancy history, metastatic disease considered.",
            
            "Consolidation in right middle lobe with air bronchograms consistent with pneumonia. "
            "Small right pleural effusion. No pneumothorax.",
            
            "Normal chest CT examination. Lungs clear without nodules, masses, or infiltrates. "
            "Heart size normal. No pericardial effusion. Mediastinum unremarkable.",
            
            "Diffuse bilateral ground-glass opacities more prominent in lower lobes. Pattern "
            "consistent with acute respiratory distress syndrome or viral pneumonitis.",
            
            "The esophagus appears normal. No mediastinal or hilar lymphadenopathy. "
            "The thyroid gland is unremarkable. No pericardial effusion is seen.",
            
            "There is no evidence of acute cardiopulmonary disease. The pulmonary vasculature "
            "is normal. No focal airspace disease. The pleural spaces are clear.",
            
            "Mild cardiomegaly noted. No pulmonary edema. The aorta is normal in caliber. "
            "The major airways are patent without evidence of obstruction.",
            
            "Calcified granuloma in the right upper lobe consistent with prior granulomatous "
            "infection. No active disease. Remainder of lungs clear.",
            
            "Small pericardial effusion noted. No tamponade physiology. Bilateral atelectasis "
            "at the lung bases. No pneumothorax or large pleural effusion.",
            
            "The left lung is clear. The right lung shows scattered nodular opacities. "
            "Lymph nodes in the mediastinum are within normal limits. The heart is not enlarged.",
            
            "Interstitial thickening is present in both lower lobes. Pleural effusions bilaterally. "
            "The trachea and main bronchi are patent. No pneumothorax identified.",
            
            "CT angiography demonstrates no filling defects in the pulmonary arteries. "
            "Subsegmental atelectasis at the lung bases. The aorta is unremarkable.",
            
            "The liver and spleen are normal in appearance. The adrenal glands are unremarkable. "
            "The kidneys are within normal limits. No abdominal lymphadenopathy.",
            
            "Coronary artery calcifications are present. The cardiac chambers are normal in size. "
            "No pericardial effusion. The ascending aorta is mildly dilated.",
        ]
    
    def _build_vocabulary(self):
        """Build vocabulary from reports."""
        cache_dir = Path(CACHE_DIR) / 'tokenizer'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f'vocab_radgenome_sentence_t{self.threshold}.pkl'
        
        # Try cache
        if cache_file.exists():
            print(f"  Loading vocabulary from: {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.token2idx = data['token2idx']
                self.idx2token = data['idx2token']
            print(f"  Vocabulary size: {len(self.token2idx)}")
            return
        
        print("  Building vocabulary from RadGenome-ChestCT Sentence column...")
        
        # Collect sentences
        sentences = self._collect_reports()
        
        # Tokenize and count
        counter = Counter()
        for sentence in sentences:
            tokens = self.tokenize_text(sentence)
            counter.update(tokens)
        
        # Build vocabulary with threshold
        self.token2idx = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        idx = len(self.special_tokens)
        
        for token, count in counter.most_common():
            if count >= self.threshold:
                self.token2idx[token] = idx
                idx += 1
        
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}
        
        print(f"  Vocabulary size: {len(self.token2idx)}")
        print(f"  (Threshold: {self.threshold}, Total unique: {len(counter)})")
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token
            }, f)
        print(f"  Saved vocabulary to: {cache_file}")
    
    def tokenize_text(self, text):
        """Tokenize text into tokens."""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.lower()
        
        # Keep measurements together
        text = re.sub(r'(\d+)\s*(mm|cm|ml|mg|gb)', r'\1\2', text)
        
        # Separate punctuation
        text = re.sub(r'([.,;:!?()])', r' \1 ', text)
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split()
        tokens = [t for t in tokens if len(t) < 50]
        
        return tokens
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs."""
        tokens = self.tokenize_text(text)
        
        ids = []
        if add_special_tokens:
            ids.append(self.sos_token_id)
        
        for token in tokens:
            ids.append(self.token2idx.get(token, self.unk_token_id))
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        if len(ids) > self.max_seq_length:
            ids = ids[:self.max_seq_length-1] + [self.eos_token_id]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        
        tokens = []
        for idx in ids:
            if isinstance(idx, (list, tuple)):
                idx = idx[0]
            idx = int(idx)
            
            token = self.idx2token.get(idx, self.unk_token)
            
            if skip_special_tokens:
                if token == self.eos_token:
                    break
                if token in self.special_tokens:
                    continue
            
            tokens.append(token)
        
        text = ' '.join(tokens)
        
        # Clean punctuation spacing
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        
        return text
    
    def __len__(self):
        return len(self.token2idx)
    
    def get_vocab_size(self):
        return len(self.token2idx)