import zipfile
import pandas as pd
import os

def solve_question(question: str, file_path: str = None) -> str:
    if "unzip" in question.lower() and file_path:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("/tmp/extracted")
        csv_file = os.path.join("/tmp/extracted", "extract.csv")
        df = pd.read_csv(csv_file)
        if "answer" in df.columns:
            return str(df["answer"].iloc[0])
        return "answer column not found"
    
    # Add more logic here for GA1–GA5 based on actual question types
    return "Logic not implemented yet"
