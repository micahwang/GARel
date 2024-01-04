import random

import argparse


import torch
from torch.optim import Adam

from util.function import garel_trainer
from generator import generator_test, generator_test_handler
from expert import ga_operator

from util.priority_queue import MaxRewardPriorityQueue
from util.dataset import SmilesCharDictionary

from tqdm import tqdm
from joblib import Parallel
from util.scoring_function import penalized_score,docking_score
import pandas as pd



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_load", type=str, default="./model/")
    parser.add_argument("--generation_steps", type=int, default=3)
    parser.add_argument("--training_steps", type=int, default=8)
    parser.add_argument("--target", type=int, default=8)

    args = parser.parse_args()

    device = torch.device(0)


    char_dict = SmilesCharDictionary()


    apprentice_storage = MaxRewardPriorityQueue()
    expert_storage = MaxRewardPriorityQueue()





    apprentice = generator_test.load(load_dir=args.generator_load)
    apprentice = apprentice.to(device)
    apprentice_optimizer = Adam(apprentice.parameters(), lr=1e-3)
    apprentice_handler = generator_test_handler(
        model=apprentice,
        optimizer=apprentice_optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=120,
    )
    apprentice.train()


    expert_handler = ga_operator(mutation_rate=0.01)

    trainer = garel_trainer(
        apprentice_storage=apprentice_storage,
        expert_storage=expert_storage,
        apprentice_handler=apprentice_handler,
        expert_handler=expert_handler,
        char_dict=char_dict,
        num_keep=1024,
        apprentice_sampling_batch_size=8196,
        expert_sampling_batch_size=8196,
        apprentice_training_batch_size=256,
        num_apprentice_training_steps=args.training_steps,
        init_smis=[],
        )


num_steps=args.generation_steps
pool = Parallel(n_jobs=8)

scoring_function=docking_score(target=args.target)


for step in tqdm(range(num_steps)):
    smis, scores = trainer.step(
        scoring_function=scoring_function, 
        device=device, 
        pool=pool
    )
    print(smis)
    if step == num_steps-1:
        df=pd.DataFrame({'SMILES':smis, 'score':scores})
        df.to_csv('out_put.csv',index=False)
