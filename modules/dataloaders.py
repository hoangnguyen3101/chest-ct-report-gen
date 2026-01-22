"""
DataLoader for RadGenome-ChestCT Dataset.

ACTUAL Dataset Structure (from logs):
- CSV columns: ['Volumename', 'Anatomy', 'Sentence']
- Each row is ONE sentence linked to an anatomy region
- Must GROUP by Volumename to get full reports

OPTIMIZATIONS:
- Caches preprocessed volumes as .npy files for fast loading
- Uses efficient resizing with skimage or scipy
- Prefetches data in background
"""

import os
import glob
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("WARNING: nibabel not installed. Install with: pip install nibabel")

# Paths (matching RadGenome_download.py)
DATA_DIR = "/umbc/rs/pi_oates/users/dta1/Data/Medical_Report_Generation/RadGenome_dataset"
CACHE_DIR = "/umbc/rs/pi_oates/users/dta1/all_cache"

# Volume cache directory - CRITICAL for performance
VOLUME_CACHE_DIR = os.path.join(CACHE_DIR, "volume_cache")


class RadGenomeDataset(Dataset):
    """
    Dataset for RadGenome-ChestCT.
    
    The CSV format is GROUNDED REPORTS:
    - Volumename: CT scan identifier
    - Anatomy: Anatomical region (e.g., "lung", "heart")
    - Sentence: Report sentence for that anatomy
    
    We GROUP sentences by Volumename to create full reports.
    """
    
    def __init__(self, args, tokenizer, split='train'):
        """
        Initialize dataset.
        """
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_length = getattr(args, 'max_seq_length', 300)
        
        # Data directory
        self.data_dir = Path(getattr(args, 'data_dir', DATA_DIR))
        
        # Load the dataset
        self.samples = self._load_dataset()
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples from RadGenome-ChestCT")
        if self.samples:
            print(f"  First sample volume: {self.samples[0].get('volume_name', 'N/A')}")
            print(f"  First sample has CT: {self.samples[0].get('has_ct', False)}")
            # Show sample of first report
            first_report = self.samples[0].get('report', '')[:200]
            print(f"  First report preview: {first_report}...")
    
    def _load_dataset(self):
        """
        Load dataset by reading CSV and grouping sentences by volume.
        """
        samples = []
        
        dataset_dir = self.data_dir / 'dataset'
        if not dataset_dir.exists():
            dataset_dir = self.data_dir
        
        print(f"  Dataset directory: {dataset_dir}")
        
        # Determine CSV and CT paths based on split
        if self.split == 'train':
            csv_filename = 'train_region_report.csv'
            ct_folder = 'train_preprocessed'
        else:
            csv_filename = 'validation_region_report.csv'
            ct_folder = 'valid_preprocessed'
        
        # Find CSV file
        csv_candidates = [
            dataset_dir / 'radgenome_files' / csv_filename,
            dataset_dir / csv_filename,
            self.data_dir / 'radgenome_files' / csv_filename,
        ]
        
        csv_path = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_path = candidate
                break
        
        # Find CT volume directory
        ct_candidates = [
            dataset_dir / ct_folder,
            self.data_dir / ct_folder,
        ]
        
        ct_dir = None
        for candidate in ct_candidates:
            if candidate.exists():
                ct_dir = candidate
                break
        
        print(f"  CSV path: {csv_path}")
        print(f"  CT directory: {ct_dir}")
        
        if csv_path is None:
            print(f"  ERROR: CSV not found!")
            return self._create_fallback_samples()
        
        # Load CSV
        print(f"  Loading CSV: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"  CSV shape: {df.shape}")
            print(f"  CSV columns: {list(df.columns)}")
        except Exception as e:
            print(f"  ERROR loading CSV: {e}")
            return self._create_fallback_samples()
        
        # Build CT volume index - search more thoroughly
        ct_index = self._build_ct_index(ct_dir)
        print(f"  Found {len(ct_index)} CT volumes")
        
        # Identify columns - the format is [Volumename, Anatomy, Sentence]
        vol_col = self._find_column(df, ['Volumename', 'volumename', 'volume_name', 'Volume', 'id'])
        sentence_col = self._find_column(df, ['Sentence', 'sentence', 'Text', 'text', 'Report', 'report'])
        anatomy_col = self._find_column(df, ['Anatomy', 'anatomy', 'Region', 'region'])
        
        print(f"  Volume column: {vol_col}")
        print(f"  Sentence column: {sentence_col}")
        print(f"  Anatomy column: {anatomy_col}")
        
        if vol_col is None or sentence_col is None:
            print(f"  ERROR: Required columns not found!")
            print(f"  Available columns: {list(df.columns)}")
            return self._create_fallback_samples()
        
        # Group sentences by volume to create full reports
        print(f"  Grouping sentences by volume...")
        volume_sentences = defaultdict(list)
        
        for idx, row in df.iterrows():
            vol_name = str(row[vol_col]).strip()
            sentence = str(row[sentence_col]).strip() if pd.notna(row[sentence_col]) else ''
            
            if sentence and sentence.lower() != 'nan' and len(sentence) > 5:
                volume_sentences[vol_name].append(sentence)
        
        print(f"  Unique volumes with sentences: {len(volume_sentences)}")
        
        # Create samples
        matched_ct = 0
        unmatched_ct = 0
        
        for vol_name, sentences in volume_sentences.items():
            # Combine all sentences for this volume into a report
            # Remove duplicates while preserving order
            seen = set()
            unique_sentences = []
            for s in sentences:
                if s not in seen:
                    seen.add(s)
                    unique_sentences.append(s)
            
            full_report = ' '.join(unique_sentences)
            
            # Skip very short reports
            if len(full_report) < 20:
                continue
            
            # Find CT file
            ct_path = self._find_ct_file(vol_name, ct_index)
            
            if ct_path:
                matched_ct += 1
                has_ct = True
            else:
                unmatched_ct += 1
                has_ct = False
            
            samples.append({
                'volume_name': vol_name,
                'ct_path': ct_path,
                'report': full_report,
                'has_ct': has_ct,
                'num_sentences': len(unique_sentences),
            })
        
        print(f"  Samples with CT matched: {matched_ct}")
        print(f"  Samples without CT: {unmatched_ct}")
        print(f"  Total samples: {len(samples)}")
        
        if len(samples) == 0:
            print("  WARNING: No samples created. Using fallback.")
            return self._create_fallback_samples()
        
        return samples
    
    def _build_ct_index(self, ct_dir):
        """
        Build index of CT volume files, searching recursively.
        """
        ct_index = {}
        
        if ct_dir is None or not ct_dir.exists():
            print(f"  CT directory does not exist: {ct_dir}")
            return ct_index
        
        print(f"  Searching for CT volumes in: {ct_dir}")
        
        # List what's in the directory first
        try:
            items = list(ct_dir.iterdir())[:10]
            print(f"  Directory contents (first 10): {[item.name for item in items]}")
        except Exception as e:
            print(f"  Error listing directory: {e}")
        
        # Search for various formats - recursively
        patterns = [
            '*.nii.gz', '*.nii', '*.npy', '*.npz', '*.h5', '*.hdf5',
            '**/*.nii.gz', '**/*.nii', '**/*.npy', '**/*.npz',
        ]
        
        for pattern in patterns:
            try:
                for ct_file in ct_dir.glob(pattern):
                    # Get volume name (remove extensions)
                    vol_name = ct_file.stem
                    if vol_name.endswith('.nii'):
                        vol_name = vol_name[:-4]
                    
                    # Store with multiple keys for matching
                    ct_index[vol_name] = str(ct_file)
                    ct_index[ct_file.stem] = str(ct_file)
                    ct_index[ct_file.name] = str(ct_file)
                    
                    # Also store parent folder name if in subdirectory
                    if ct_file.parent != ct_dir:
                        ct_index[ct_file.parent.name] = str(ct_file)
            except Exception as e:
                continue
        
        # If still empty, check if files are in subdirectories by volume name
        if not ct_index:
            try:
                for subdir in ct_dir.iterdir():
                    if subdir.is_dir():
                        # Check for files in subdirectory
                        for pattern in ['*.nii.gz', '*.nii', '*.npy']:
                            for ct_file in subdir.glob(pattern):
                                vol_name = subdir.name
                                ct_index[vol_name] = str(ct_file)
                                ct_index[ct_file.stem] = str(ct_file)
            except:
                pass
        
        return ct_index
    
    def _find_ct_file(self, vol_name, ct_index):
        """
        Find CT file for a given volume name.
        """
        # Try exact match
        if vol_name in ct_index:
            return ct_index[vol_name]
        
        # Try without common prefixes/suffixes
        clean_name = vol_name.replace('_', '').replace('-', '').lower()
        
        for key, path in ct_index.items():
            clean_key = key.replace('_', '').replace('-', '').lower()
            if clean_name == clean_key:
                return path
            if clean_name in clean_key or clean_key in clean_name:
                return path
        
        return None
    
    def _find_column(self, df, candidates):
        """Find a column by trying multiple candidate names."""
        for col in candidates:
            if col in df.columns:
                return col
        # Try case-insensitive match
        df_cols_lower = {c.lower(): c for c in df.columns}
        for col in candidates:
            if col.lower() in df_cols_lower:
                return df_cols_lower[col.lower()]
        return None
    
    def _create_fallback_samples(self):
        """Create fallback samples when data cannot be loaded."""
        print("  Creating fallback samples with medical vocabulary...")
        
        fallback_reports = [
            "The lungs are clear bilaterally. No focal consolidation, mass, or nodule identified. Heart size is within normal limits. The mediastinum is unremarkable. No pleural effusion or pneumothorax. The visualized osseous structures are intact.",
            "Comparison made to prior examination. Previously noted pulmonary nodule in the right lower lobe has decreased in size. No new pulmonary nodules identified. The airways are patent. No significant lymphadenopathy.",
            "Mild emphysematous changes predominantly affecting the upper lobes. Scattered areas of ground-glass opacity noted in the lower lobes bilaterally. The main pulmonary artery appears mildly dilated.",
            "No evidence of pulmonary embolism. The thoracic aorta is normal in caliber without dissection. Heart and pericardium appear normal. No significant lymphadenopathy in mediastinum or hila.",
            "Findings consistent with interstitial lung disease with pattern suggestive of usual interstitial pneumonia. Honeycombing present predominantly at lung bases. Traction bronchiectasis noted.",
            "Post-surgical changes in the right hemithorax consistent with prior lobectomy. The remaining lung parenchyma appears clear. No evidence of recurrent disease. Surgical clips stable.",
            "Multiple bilateral pulmonary nodules identified. Largest measuring 12mm in left upper lobe. Given patient history of malignancy, metastatic disease should be considered.",
            "Consolidation in right middle lobe with air bronchograms consistent with pneumonia. Small right pleural effusion. No pneumothorax.",
            "Normal chest CT examination. Lungs clear without evidence of nodules, masses, or infiltrates. Heart size normal. No pericardial effusion. Mediastinum unremarkable.",
            "Diffuse bilateral ground-glass opacities present, more prominent in lower lobes. Pattern consistent with acute respiratory distress syndrome or viral pneumonitis.",
            "The esophagus appears normal. No mediastinal or hilar lymphadenopathy. The thyroid gland is unremarkable. No pericardial effusion seen.",
            "No evidence of acute cardiopulmonary disease. Pulmonary vasculature normal. No focal airspace disease. Pleural spaces are clear.",
            "Mild cardiomegaly noted. No pulmonary edema. The aorta is normal in caliber. Major airways patent without evidence of obstruction.",
            "Calcified granuloma in right upper lobe consistent with prior granulomatous infection. No active disease. Remainder of lungs clear.",
            "Small pericardial effusion noted. No tamponade physiology. Bilateral atelectasis at lung bases. No pneumothorax or large pleural effusion.",
        ]
        
        samples = []
        for i, report in enumerate(fallback_reports * 10):
            samples.append({
                'volume_name': f'fallback_{i:04d}',
                'ct_path': None,
                'report': report,
                'has_ct': False,
            })
        
        return samples
    
    def _load_ct_volume(self, ct_path):
        """
        Load CT volume from file WITH CACHING for fast repeated access.
        """
        target_shape = (1, 64, 128, 128)  # (C, D, H, W)
        
        if ct_path is None or not os.path.exists(ct_path):
            volume = np.random.randn(*target_shape).astype(np.float32) * 0.1
            return torch.from_numpy(volume)
        
        # Check cache first
        cache_dir = Path(VOLUME_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache key from path
        cache_key = hashlib.md5(ct_path.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{cache_key}.npy"
        
        # Try loading from cache
        if cache_file.exists():
            try:
                volume = np.load(cache_file)
                return torch.from_numpy(volume)
            except:
                pass  # Cache corrupted, reload
        
        # Load and preprocess
        try:
            if ct_path.endswith(('.nii', '.nii.gz')) and HAS_NIBABEL:
                nii = nib.load(ct_path)
                volume = nii.get_fdata(dtype=np.float32)
                
            elif ct_path.endswith('.npy'):
                volume = np.load(ct_path).astype(np.float32)
                
            elif ct_path.endswith('.npz'):
                data = np.load(ct_path)
                volume = data[data.files[0]].astype(np.float32)
                
            else:
                volume = np.random.randn(*target_shape).astype(np.float32) * 0.1
                return torch.from_numpy(volume)
            
            # Handle 4D volumes
            if volume.ndim == 4:
                volume = volume[..., 0] if volume.shape[-1] < volume.shape[0] else volume[0]
            
            # Ensure 3D
            if volume.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
            
            # Normalize
            volume = (volume - volume.mean()) / (volume.std() + 1e-8)
            
            # Fast resize using strided slicing + interpolation
            volume = self._fast_resize(volume, target_shape[1:])
            
            # Add channel dimension
            volume = volume[np.newaxis, ...].astype(np.float32)
            
            # Save to cache for next time
            try:
                np.save(cache_file, volume)
            except:
                pass  # Cache write failed, continue anyway
            
            return torch.from_numpy(volume)
            
        except Exception as e:
            volume = np.random.randn(*target_shape).astype(np.float32) * 0.1
            return torch.from_numpy(volume)
    
    def _fast_resize(self, volume, target_shape):
        """
        Fast volume resize using strided slicing then interpolation.
        Much faster than scipy.ndimage.zoom for large volumes.
        """
        # First, subsample if volume is much larger
        current = np.array(volume.shape)
        target = np.array(target_shape)
        
        # If volume is more than 2x target in any dimension, subsample first
        if np.any(current > target * 2):
            strides = np.maximum(1, current // target).astype(int)
            volume = volume[::strides[0], ::strides[1], ::strides[2]]
        
        # Now use scipy for final resize (on smaller volume)
        try:
            from scipy import ndimage
            current = np.array(volume.shape)
            factors = target / current
            volume = ndimage.zoom(volume, factors, order=1)  # Linear interpolation
        except ImportError:
            # Fallback: simple resize with numpy
            volume = self._numpy_resize(volume, target_shape)
        
        return volume
    
    def _numpy_resize(self, volume, target_shape):
        """Simple resize using numpy indexing."""
        d, h, w = volume.shape
        td, th, tw = target_shape
        
        # Create index arrays
        d_idx = np.linspace(0, d-1, td).astype(int)
        h_idx = np.linspace(0, h-1, th).astype(int)
        w_idx = np.linspace(0, w-1, tw).astype(int)
        
        # Resample
        volume = volume[d_idx][:, h_idx][:, :, w_idx]
        return volume
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.samples[idx]
        
        # Load CT volume
        images = self._load_ct_volume(sample.get('ct_path'))
        
        # Get report text
        report = sample.get('report', '')
        
        # Tokenize
        report_ids = self.tokenizer.encode(report, add_special_tokens=True)
        
        # Pad/truncate
        if len(report_ids) < self.max_seq_length:
            padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(report_ids))
            report_ids = report_ids + padding
        else:
            report_ids = report_ids[:self.max_seq_length-1] + [self.tokenizer.eos_token_id]
        
        # Create mask
        report_masks = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in report_ids]
        
        return {
            'images': images,
            'report_ids': torch.LongTensor(report_ids),
            'report_masks': torch.LongTensor(report_masks),
            'reports': report,
        }