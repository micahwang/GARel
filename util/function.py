from time import time
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.nn as nn
from rdkit import Chem
import random

def canonicalize(smiles: str, include_stereocenters=True):

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None


def process_smis(
    smis,
    scoring_function,
    pool,
    canonicalization,
    duplicate_removal,
    scoring_parallelization,
    max_smi_len=100,
):
    if canonicalization:
        smis = pool(
            delayed(lambda smi: canonicalize(smi, include_stereocenters=False))(smi) for smi in smis
        )
        smis = list(filter(lambda smi: (smi is not None) and (len(smi) < max_smi_len), smis))

    if duplicate_removal:
        smis = list(set(smis))

    if scoring_function is None:
        return smis

    if scoring_parallelization:
        scores = pool(delayed(scoring_function)(smi) for smi in smis)
    else:
        scores = [scoring_function(smi) for smi in smis]

    smis, scores = filter_by_score(smis, scores, -1e-8)

    return smis, scores


def smis_to_actions(char_dict, smis):
    max_seq_length = 121
    enc_smis = list(map(lambda smi: char_dict.encode(smi) + char_dict.END, smis))
    actions = np.zeros((len(smis), max_seq_length), dtype=np.int32)
    seq_lengths = np.zeros((len(smis),), dtype=np.long)

    for i, enc_smi in list(enumerate(enc_smis)):
        for c in range(len(enc_smi)):
            try:
                actions[i, c] = char_dict.char_idx[enc_smi[c]]
            except:
                print(char_dict.char_idx)
                print(enc_smi)
                print(enc_smi[c])
                assert False

        seq_lengths[i] = len(enc_smi)

    return actions, seq_lengths


def filter_by_score(smis, scores, score_thr):
    filtered_smis_and_scores = list(filter(lambda elem: elem[1] > score_thr, zip(smis, scores)))
    filtered_smis, filtered_scores = map(list, zip(*filtered_smis_and_scores))
    return filtered_smis, filtered_scores


class LatentLoss(nn.Module):    
    def __init__(self):
        super(LatentLoss, self).__init__()

    def forward(self, mu, logvar):
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()


        return kl_loss





class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, re_x, x):
        diffs = torch.add(x, -re_x)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, re_x, x):
        diffs = torch.add(x, -re_x)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, x, y):

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        y = y.view(batch_size, -1)

        x_l2_norm = torch.norm(x, p=2, dim=1, keepdim=True).detach()
        x_l2 = x.div(x_l2_norm.expand_as(x) + 1e-6)

        y_l2_norm = torch.norm(y, p=2, dim=1, keepdim=True).detach()
        y_l2 = y.div(y_l2_norm.expand_as(y) + 1e-6)

        diff_loss = torch.mean((x_l2.t().mm(y_l2)).pow(2))

        return diff_loss

class SimLoss(nn.Module):    
    def __init__(self):
        super(SimLoss, self).__init__()


    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def forward(self,x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

class garel_trainer:
    def __init__(
        self,
        apprentice_storage,
        expert_storage,
        apprentice_handler,
        expert_handler,
        char_dict,
        num_keep,
        apprentice_sampling_batch_size,
        expert_sampling_batch_size,
        apprentice_training_batch_size,
        num_apprentice_training_steps,
        init_smis,
    ):
        self.apprentice_storage = apprentice_storage
        self.expert_storage = expert_storage

        self.apprentice_handler = apprentice_handler
        self.expert_handler = expert_handler

        self.char_dict = char_dict

        self.num_keep = num_keep
        self.apprentice_sampling_batch_size = apprentice_sampling_batch_size
        self.expert_sampling_batch_size = expert_sampling_batch_size
        self.apprentice_training_batch_size = apprentice_training_batch_size
        self.num_apprentice_training_steps = num_apprentice_training_steps

        self.init_smis = init_smis

    def init(self, scoring_function, device, pool):
        if len(self.init_smis) > 0:
            smis, scores = self.canonicalize_and_score_smiles(
                smis=self.init_smis, scoring_function=scoring_function, pool=pool
            )

            self.apprentice_storage.add_list(smis=smis, scores=scores)
            self.expert_storage.add_list(smis=smis, scores=scores)

    def step(self, scoring_function, device, pool):
        apprentice_smis, apprentice_scores = self.update_storage_by_apprentice(
            scoring_function, device, pool
        )
        expert_smis, expert_scores = self.update_storage_by_expert(scoring_function, pool)
        loss, fit_size = self.train_apprentice_step(device)


        return apprentice_smis + expert_smis, apprentice_scores + expert_scores

    def update_storage_by_apprentice(self, scoring_function, device, pool):
        with torch.no_grad():
            self.apprentice_handler.model.eval()
            smis, _, _, _ = self.apprentice_handler.sample(
                num_samples=self.apprentice_sampling_batch_size, device=device
            )
        smis, scores = self.canonicalize_and_score_smiles(
            smis=smis, scoring_function=scoring_function, pool=pool
        )

        self.apprentice_storage.add_list(smis=smis, scores=scores)
        self.apprentice_storage.squeeze_by_kth(k=self.num_keep)

        return smis, scores

    def update_storage_by_expert(self, scoring_function, pool):
        expert_smis, expert_scores = self.apprentice_storage.sample_batch(
            self.expert_sampling_batch_size
        )
        smis = self.expert_handler.query(
            query_size=self.expert_sampling_batch_size, mating_pool=expert_smis, pool=pool
        )
        smis, scores = self.canonicalize_and_score_smiles(
            smis=smis, scoring_function=scoring_function, pool=pool
        )

        self.expert_storage.add_list(smis=smis, scores=scores)
        self.expert_storage.squeeze_by_kth(k=self.num_keep)

        return smis, scores

    def train_apprentice_step(self, device):
        avg_loss = 0.0
        apprentice_smis, _ = self.apprentice_storage.get_elems()
        expert_smis, _ = self.expert_storage.get_elems()
        total_smis = list(set(apprentice_smis + expert_smis))

        self.apprentice_handler.model.train()
        for _ in range(self.num_apprentice_training_steps):
            smis = random.choices(population=total_smis, k=self.apprentice_training_batch_size)
            loss = self.apprentice_handler.train_on_batch(smis=smis, device=device)

            avg_loss += loss / self.num_apprentice_training_steps

        fit_size = len(total_smis)

        return avg_loss, fit_size

    def canonicalize_and_score_smiles(self, smis, scoring_function, pool):
        smis = pool(
            delayed(lambda smi: canonicalize(smi, include_stereocenters=False))(smi) for smi in smis
        )
        smis = list(filter(lambda smi: (smi is not None) and self.char_dict.allowed(smi), smis))
        scores = pool(delayed(scoring_function.score)(smi) for smi in smis)
        # scores = [0.0 for smi in smis]

        filtered_smis_and_scores = list(
            filter(
                lambda smi_and_score: smi_and_score[1]
                > -999,
                #> scoring_function.scoring_function.corrupt_score,
                zip(smis, scores),
            )
        )

        smis, scores = (
            map(list, zip(*filtered_smis_and_scores))
            if len(filtered_smis_and_scores) > 0
            else ([], [])
        )
        return smis, scores
