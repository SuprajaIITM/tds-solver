import os
import tempfile
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

def unzip_folder(file):
    with ZipFile(BytesIO(file.file.read())) as zip_ref:
        temp_dir = tempfile.mkdtemp()
        extracted_files = {}

        for zip_info in zip_ref.infolist():
            extracted_path = zip_ref.extract(zip_info, path=temp_dir)
            with open(extracted_path, "rb") as f:
                extracted_files[zip_info.filename] = BytesIO(f.read())

        return temp_dir, extracted_files