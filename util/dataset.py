import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np




PAD = " "
BEGIN = "Q"
END = "\n"

def load_dataset(char_dict, smi_path):
    processed_dataset_path = str(Path(smi_path).with_suffix("")) + "_processed.smiles"
    if False and os.path.exists(processed_dataset_path):
        with open(processed_dataset_path, "r") as f:
            processed_dataset = f.readlines()

    else:
        with open(smi_path, "r") as f:
            dataset = f.read().splitlines()

        processed_dataset = list(filter(char_dict.allowed, dataset))
        with open(processed_dataset_path, "w") as f:
            f.write("\n".join(processed_dataset))

    return processed_dataset


class SmilesCharDictionary(object):
    def __init__(self) -> None:
        


        self.char_idx = {PAD: 0,BEGIN: 1,END: 2,"#": 20,"%": 22,"(": 25,")": 24,"+": 26, "-": 27, ".": 30,"0": 32,"1": 31,
"2": 34,"3": 33,"4": 36,"5": 35,"6": 38,"7": 37,"8": 40,"9": 39,"=": 41,"A": 7,"B": 11,"C": 19,"F": 4,"H": 6,"I": 5,"N": 10,
"O": 9,"P": 12,"S": 13,"X": 15,"Y": 14,"Z": 3,"[": 16,"]": 18,"b": 21,"c": 8,"n": 17,"o": 29,"p": 23,"s": 28,"@": 42,"R": 43,
"/": 44,"\\": 45,"E": 46,}

        self.idx_char = {v: k for k, v in self.char_idx.items()}

        self.encode_dict = {"Br": "Y", "Cl": "X", "Si": "A", "Se": "Z", "@@": "R", "se": "E"}
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    def allowed(self, smiles) -> bool:
        if len(smiles) > 120:
            return False
        return True

    def encode(self, smiles: str) -> str:
        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def decode(self, smiles):
        temp_smiles = smiles
        for symbol, token in self.decode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)

        return temp_smiles

    def get_char_num(self) -> int:
        return len(self.idx_char)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[BEGIN]

    @property
    def end_idx(self) -> int:
        return self.char_idx[END]

    @property
    def pad_idx(self) -> int:
        return self.char_idx[PAD]

    @property
    def BEGIN(self):
        return BEGIN

    @property
    def END(self):
        return END

    @property
    def PAD(self):
        return PAD

    def matrix_to_smiles(self, array, seq_lengths):
        array = array.tolist()
        smis = list(
            map(lambda item: self.vector_to_smiles(item[0], item[1]), zip(array, seq_lengths))
        )
        return smis

    def vector_to_smiles(self, vec, seq_length):
        chars = list(map(self.idx_char.get, vec[:seq_length]))
        smi = "".join(chars)
        smi = self.decode(smi)
        return smi

