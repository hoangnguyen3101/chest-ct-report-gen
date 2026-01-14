import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]


class CTReportDataset(Dataset):
    def __init__(self, args, data_folder, xlsx_file, tokenizer, min_slices=20, resize_dim=500, num_frames=2, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.accession_to_text = self.load_accession_text(xlsx_file)
        self.paths=[]
        self.samples = self.prepare_samples()

        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    # def load_accession_text(self, csv_file):
    #     df = pd.read_csv(csv_file)
    #     accession_to_text = {}
    #     for index, row in df.iterrows():
    #         key = row.get('Volumename', None)
    #         if pd.isna(key):
    #             continue
    #         val = row.get('Sentence', "")
    #         if pd.isna(val):
    #             val = ""
    #         else:
    #             val = str(val)
    #         accession_to_text[str(key)] = val
    #     return accession_to_text

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        df.columns = [c.strip() for c in df.columns] # loại bỏ khoảng thừa
        
        accession_to_text = {}
        
        # Check if Volumename exists
        if 'Volumename' not in df.columns:
            print(f"ERROR: 'Volumename' column not found. Available: {list(df.columns)}")
            return {}

        # Fill NaNs
        df = df.fillna('')
        
        grouped = df.groupby('Volumename') #gom nhóm các hàng có chung tên file ảnh
        for name, group in grouped:
            report_parts = []
            for index, row in group.iterrows():
                try:
                    anatomy = str(row.get('Anatomy', '')).strip() # lấy tên vùng cơ thể
                    sentence = str(row.get('Sentence', '')).strip() # lấy mô tả vùng cơ thể
                    

                    if anatomy and sentence: 
                        report_parts.append(f"{anatomy} {sentence}") # nếu có cả 2 thì  nối lại theo dạng  phổi: mô tả phổi
                    elif sentence:
                        report_parts.append(sentence)
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
            
            # Combine all parts into one single string
            if report_parts:
                accession_to_text[str(name)] = ". ".join(report_parts)
                
        return accession_to_text


    def prepare_samples(self):
        samples = []
        
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                    filename = os.path.basename(nii_file)
                    if filename not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[filename]

                    # Construct the input text with the included metadata
                    if impression_text == "Not given.":
                        impression_text=""

                    input_text_concat = str(impression_text)

                    input_text = f'{input_text_concat}'
                    samples.append((nii_file, input_text_concat))
                    self.paths.append(nii_file)

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        img_data = nib.load(path).get_fdata()
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data+400 ) / 600)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)

        # Get the dimensions of the input tensor
        target_shape = (480,480,240)
        
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)
        
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor[0]

    
    def __getitem__(self, index):
        img, text = self.samples[index]
        img_id = img.split("/")[-1]
        tensor = self.nii_to_tensor(img)
        ids = self.tokenizer(text)[:self.max_seq_length]
        mask = [1] * len(ids)
        seq_lenght = len(ids)
        sample = (img_id, tensor, ids, mask, seq_lenght)
        return sample
