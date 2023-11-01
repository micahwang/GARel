import os
import json

import torch
import torch.nn as nn
from torch.distributions import Categorical
from util.function import smis_to_actions
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from tkinter import S
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from util.function import LatentLoss, DiffLoss, SimLoss
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import torch.nn.functional as F

import argparse



class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.rl1 = nn.ReLU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.rl2 = nn.ReLU()

        out_channels *= 2
        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.rl1(x)

        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x2 = self.rl2(x1)

        return x2



class Block_Pool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.rl1 = nn.ReLU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.rl2 = nn.ReLU()

        out_channels *= 2
        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.rl1(x)

        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x2 = self.rl2(x1)

        return x2






class generator(nn.Module):
    def __init__(self,in_channel_0, out_channel_0,in_channel_1,out_channel_1,in_channel_2,out_channel_2,
                 in_channel_3,out_channel_3):
        super(generator).__init__()
        self.in_channel_0=in_channel_0

        self.out_channel_0=out_channel_0
        self.in_channel_1=in_channel_1
        self.out_channel_1=out_channel_1
        self.in_channel_2=in_channel_2
        self.out_channel_2=out_channel_2
        self.in_channel_3=in_channel_3
        self.out_channel_3=out_channel_3
        # private source encoder(Ligand in ZINC dataset)
        shr_enc = []
        self.fc11 = nn.Linear(256, 512)
        self.fc12 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 39)
        self.gru1 = nn.GRU(39, 256, 3, batch_first=True)
        self.embedding = nn.Embedding(39, 39, 0)
        self.gru = nn.GRU(551, 1024, 3, batch_first=True)

        # pri_source_encoder 

        self.blocks_1 = nn.Sequential(*[Block(in_channel_0, out_channel_0), Block_Pool(out_channel_0, in_channel_1)])
        self.blocks_2 = nn.Sequential(*[Block(in_channel_1, in_channel_1), Block_Pool(in_channel_1, out_channel_1)])
        self.blocks3 = nn.Sequential(*[Block(in_channel_2, out_channel_1), Block_Pool(out_channel_1, out_channel_2)])
        self.blocks4 = nn.Sequential(*[Block(in_channel_3, out_channel_2), Block(out_channel_2, out_channel_3)])
    
        # pri_target_encoder


        self.blocks_1_1 = nn.Sequential(*[Block(in_channel_0, out_channel_0), Block_Pool(out_channel_0, in_channel_1)])
        self.blocks_2_2 = nn.Sequential(*[Block(in_channel_1, in_channel_1), Block_Pool(in_channel_1, out_channel_1)])
        self.blocks_3_3 = nn.Sequential(*[Block(in_channel_2, out_channel_1), Block_Pool(out_channel_1, out_channel_2)])
        self.blocks_4_4 = nn.Sequential(*[Block(in_channel_3, out_channel_2), Block(out_channel_2, out_channel_3)])

        self.blocks_1_1_1 = nn.Sequential(*[Block(in_channel_0, out_channel_0), Block_Pool(out_channel_0, in_channel_1)])
        self.blocks_2_2_2 = nn.Sequential(*[Block(in_channel_1, in_channel_1), Block_Pool(in_channel_1, out_channel_1)])
        self.blocks_3_3_3 = nn.Sequential(*[Block(in_channel_2, out_channel_1), Block_Pool(out_channel_1, out_channel_2)])
        self.blocks_4_4_4 = nn.Sequential(*[Block(in_channel_3, out_channel_2), Block(out_channel_2, out_channel_3)])











    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        eps = Variable(eps).cuda()
        return (eps.mul(std)).add_(mu)

    def decode(self, z, captions):
        captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
        lengths = [len(i_x) for i_x in captions]
        x_emb = self.embedding(captions)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = pack_padded_sequence(x_input, lengths, batch_first=True)
        h_0 = self.fc3(z)
        h_0 = h_0.unsqueeze(0).repeat(3, 1, 1)
        output, _ = self.gru(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.fc4(output)
        return y

    def forward(self, input_data, input_caption, mode, rec_scheme):
        result = []
  
        if mode == 'source':
            # source private encoder

            x = self.blocks_1(input_data)
            x = torch.cat([self.blocks_2(x), x], dim=1)
            x = torch.cat([self.blocks3(x), x], dim=1)
            x = self.blocks4(x) 


            x = x.mean(2).mean(2).mean(2)
            pri_feat_mu, pri_feat_logvar = self.fc11(x), self.fc12(x)
            pri_latent = self.reparametrize(pri_feat_mu, pri_feat_logvar)

            x_cap = [self.embedding(i_c.cuda()) for i_c in input_caption]
            x_cap = nn.utils.rnn.pack_sequence(x_cap, enforce_sorted=False)
            _, h = self.gru1(x_cap)
            h = h[-(1 + int(self.gru1.bidirectional)):]
            h = torch.cat(h.split(1), dim=-1).squeeze(0)
            cap_mu, cap_logvar = self.fc11(h), self.fc12(h)
            cap_pri_latent = self.reparametrize(cap_mu, cap_logvar)
            
        elif mode == 'target':

            # target private encoder
            # x = pri_source_encoder(input_data)
            # [1,52,16,16,16]
            x = self.blocks_1_1(input_data)
            x = torch.cat([self.blocks_2_2(x), x], dim=1)
            x = torch.cat([self.blocks_3_3(x), x], dim=1)
            x = self.blocks44(x) 
            #  ==========修改的地方===============
            
            x = x.mean(2).mean(2).mean(2)
            pri_feat_mu, pri_feat_logvar = self.fc11(x), self.fc12(x)
            pri_latent = self.reparametrize(pri_feat_mu, pri_feat_logvar)

            x_cap = [self.embedding(i_c.cuda()) for i_c in input_caption]
            x_cap = nn.utils.rnn.pack_sequence(x_cap, enforce_sorted=False)
            _, h = self.gru1(x_cap)
            h = h[-(1 + int(self.gru1.bidirectional)):]
            h = torch.cat(h.split(1), dim=-1).squeeze(0)
            cap_mu, cap_logvar = self.fc11(h), self.fc12(h)
            cap_pri_latent = self.reparametrize(cap_mu, cap_logvar)
        result.extend([pri_feat_mu, pri_feat_logvar, pri_latent, cap_mu, cap_logvar, cap_pri_latent])
        
        
        
        
        # shared encoder
        x = self.blocks_1_1_1(input_data)
        x = torch.cat([self.blocks_2_2_2(x), x], dim=1)
        x = torch.cat([self.blocks_3_3_3(x), x], dim=1)
        x = self.blocks_4_4_4(x)

        x = x.mean(2).mean(2).mean(2)
        shr_feat_mu, shr_feat_logvar = self.fc11(x), self.fc12(x)
        shr_latent = self.reparametrize(shr_feat_mu, shr_feat_logvar)



        x_cap = [self.embedding(i_c.cuda()) for i_c in input_caption]
        x_cap = nn.utils.rnn.pack_sequence(x_cap, enforce_sorted=False)
        _, h = self.gru1(x_cap)
        h = h[-(1 + int(self.gru1.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        cap_mu, cap_logvar = self.fc11(h), self.fc12(h)
        cap_shr_latent = self.reparametrize(cap_mu, cap_logvar)

        result.extend([shr_latent, cap_shr_latent])

        # shared decoder

        if rec_scheme == 'share':
            union_latent = cap_shr_latent
        elif rec_scheme == 'all':
            union_latent = cap_shr_latent + cap_pri_latent
        elif rec_scheme == 'private':
            union_latent = cap_pri_latent

        # re_smi = self.decode(cap_pri_latent,caption,lengths)
        cap = [i_c.cuda() for i_c in input_caption]
        re_smi = self.decode(union_latent, cap)
        result.append(re_smi)

        return result
    def load(cls, load_dir):
        model_config_path = os.path.join(load_dir, "generator1_config.json")
        with open(model_config_path, "r") as f:
            config = json.load(f)

        model = cls(**config)

        model_weight_path = os.path.join(load_dir, "generator1_weight.pt")
        try:
            model_state_dict = torch.load(model_weight_path, map_location="cpu")
            model.load_state_dict(model_state_dict)
        except:
            print("No pretrained weight for SmilesGenerator.")

        return model

    def save(self, save_dir):
        model_config = self.config
        model_config_path = os.path.join(save_dir, "generator_config1.json")
        with open(model_config_path, "w") as f:
            json.dump(model_config, f)

        model_state_dict = self.state_dict()
        model_weight_path = os.path.join(save_dir, "generator1_weight.pt")
        torch.save(model_state_dict, model_weight_path)

    @property
    def config(self):
        return dict(
                in_channel_0=self.in_channel_0, 
                out_channel_0=self.in_channel_0,
                in_channel_1=self.in_channel_1,
                out_channel_1=self.in_channel_1,
                in_channel_2=self.in_channel_2,
                out_channel_2=self.in_channel_2,
                in_channel_3=self.in_channel_3,
                out_channel_3=self.in_channel_3
        )


class generator_handler:
    def __init__(self, model,char_dict,max_sampling_batch_size,optimizer):
        self.model = model
        self.max_sampling_batch_size = max_sampling_batch_size
        self.char_dict = char_dict
        self.max_seq_length = 121
        
    def sample(self, num_samples, device):
        action, log_prob, seq_length = self.sample_action(num_samples=num_samples, device=device)
        smiles = self.char_dict.matrix_to_smiles(action, seq_length - 1)

        return smiles, action, log_prob, seq_length

    def sample_action(self, num_samples, device):
        number_batches = (
            num_samples + self.max_sampling_batch_size - 1
        ) // self.max_sampling_batch_size
        remaining_samples = num_samples

        action = torch.LongTensor(num_samples, self.max_seq_length).to(device)
        log_prob = torch.FloatTensor(num_samples, self.max_seq_length).to(device)
        seq_length = torch.LongTensor(num_samples).to(device)

        batch_start = 0

        for i in range(number_batches):
            batch_size = min(self.max_sampling_batch_size, remaining_samples)
            batch_end = batch_start + batch_size

            action_batch, log_prob_batch, seq_length_batch = self._sample_action_batch(
                batch_size, device
            )
            action[batch_start:batch_end, :] = action_batch
            log_prob[batch_start:batch_end, :] = log_prob_batch
            seq_length[batch_start:batch_end] = seq_length_batch

            batch_start += batch_size
            remaining_samples -= batch_size

        return action, log_prob, seq_length

    def train_on_batch(self, smis, device, weights=1.0):
        actions, _ = smis_to_actions(self.char_dict, smis)
        actions = torch.LongTensor(actions)
        loss = self.train_on_action_batch(actions=actions, device=device, weights=weights)
        return loss

    def train_on_action_batch(self, actions, target, device, weights=1.0):




        batch_size = actions.size(0)
        batch_seq_length = actions.size(1)

        actions = actions.to(device)

        start_token_vector = self._get_start_token_vector(batch_size, device)
        input_actions = torch.cat([start_token_vector, actions[:, :-1]], dim=1)
        target_actions = actions

        input_actions = input_actions.to(device)
        target_actions = target_actions.to(device)



        output, _ = self.model(input_actions, hidden=None)
        output = output.view(batch_size * batch_seq_length, -1)

        log_probs = torch.log_softmax(output, dim=1)
        log_target_probs = log_probs.gather(dim=1, index=target_actions.reshape(-1, 1)).squeeze(
            dim=1
        )
        log_target_probs = log_target_probs.view(batch_size, batch_seq_length).mean(dim=1)


        loss = -(weights * log_target_probs).mean()

        if self.entropy_factor > 0.0:
            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=1).mean()
            loss -= self.entropy_factor * entropy

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def get_action_log_prob(self, actions, seq_lengths, device):
        num_samples = actions.size(0)
        actions_seq_length = actions.size(1)
        log_probs = torch.FloatTensor(num_samples, actions_seq_length).to(device)

        number_batches = (
            num_samples + self.max_sampling_batch_size - 1
        ) // self.max_sampling_batch_size
        remaining_samples = num_samples
        batch_start = 0
        for i in range(number_batches):
            batch_size = min(self.max_sampling_batch_size, remaining_samples)
            batch_end = batch_start + batch_size
            log_probs[batch_start:batch_end, :] = self._get_action_log_prob_batch(
                actions[batch_start:batch_end, :], seq_lengths[batch_start:batch_end], device
            )
            batch_start += batch_size
            remaining_samples -= batch_size

        return log_probs

    def save(self, save_dir):
        self.model.save(save_dir)

    def _get_action_log_prob_batch(self, actions, seq_lengths, device):
        batch_size = actions.size(0)
        actions_seq_length = actions.size(1)

        start_token_vector = self._get_start_token_vector(batch_size, device)
        input_actions = torch.cat([start_token_vector, actions[:, :-1]], dim=1)
        target_actions = actions

        input_actions = input_actions.to(device)
        target_actions = target_actions.to(device)

        output, _ = self.model(input_actions, hidden=None)
        output = output.view(batch_size * actions_seq_length, -1)
        log_probs = torch.log_softmax(output, dim=1)
        log_target_probs = log_probs.gather(dim=1, index=target_actions.reshape(-1, 1)).squeeze(
            dim=1
        )
        log_target_probs = log_target_probs.view(batch_size, self.max_seq_length)

        mask = torch.arange(actions_seq_length).expand(len(seq_lengths), actions_seq_length) > (
            seq_lengths - 1
        ).unsqueeze(1)
        log_target_probs[mask] = 0.0

        return log_target_probs

    def _sample_action_batch(self, batch_size, device):
        hidden = None
        inp = self._get_start_token_vector(batch_size, device)

        action = torch.zeros((batch_size, self.max_seq_length), dtype=torch.long).to(device)
        log_prob = torch.zeros((batch_size, self.max_seq_length), dtype=torch.float).to(device)
        seq_length = torch.zeros(batch_size, dtype=torch.long).to(device)

        ended = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for t in range(self.max_seq_length):
            output, hidden = self.model(inp, hidden)

            prob = torch.softmax(output, dim=2)
            distribution = Categorical(probs=prob)
            action_t = distribution.sample()
            log_prob_t = distribution.log_prob(action_t)
            inp = action_t

            action[~ended, t] = action_t.squeeze(dim=1)[~ended]
            log_prob[~ended, t] = log_prob_t.squeeze(dim=1)[~ended]

            seq_length += (~ended).long()
            ended = ended | (action_t.squeeze(dim=1) == self.char_dict.end_idx).bool()

            if ended.all():
                break

        return action, log_prob, seq_length

    def _get_start_token_vector(self, batch_size, device):
        return torch.LongTensor(batch_size, 1).fill_(self.char_dict.begin_idx).to(device)








class generator_test(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, lstm_dropout):
        super(generator_test, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.lstm_dropout = lstm_dropout

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=lstm_dropout
        )
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)

        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0)

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        
                r_gate = param[int(0.25 * len(param)) : int(0.5 * len(param))]
                nn.init.constant_(r_gate, 1)

    def forward(self, x, hidden):
        embeds = self.encoder(x)
        output, hidden = self.lstm(embeds, hidden)
        output = self.decoder(output)
        return output, hidden

    @classmethod
    def load(cls, load_dir):
        model_config_path = os.path.join(load_dir, "generator_config.json")
        with open(model_config_path, "r") as f:
            config = json.load(f)

        model = cls(**config)

        model_weight_path = os.path.join(load_dir, "generator_weight.pt")
        try:
            model_state_dict = torch.load(model_weight_path, map_location="cpu")
            rnn_keys = [key for key in model_state_dict if key.startswith("rnn")]
            for key in rnn_keys:
                weight = model_state_dict.pop(key)
                model_state_dict[key.replace("rnn", "lstm")] = weight

            model.load_state_dict(model_state_dict)
        except:
            print("No pretrained weight for SmilesGenerator.")

        return model

    def save(self, save_dir):
        model_config = self.config
        model_config_path = os.path.join(save_dir, "generator_config.json")
        with open(model_config_path, "w") as f:
            json.dump(model_config, f)

        model_state_dict = self.state_dict()
        model_weight_path = os.path.join(save_dir, "generator_weight.pt")
        torch.save(model_state_dict, model_weight_path)

    @property
    def config(self):
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            lstm_dropout=self.lstm_dropout,
        )


class generator_test_handler:
    def __init__(self, model, optimizer, char_dict, max_sampling_batch_size, entropy_factor=0.0):
        self.model = model
        self.optimizer = optimizer
        self.max_sampling_batch_size = max_sampling_batch_size
        self.entropy_factor = entropy_factor
        self.char_dict = char_dict
        self.max_seq_length = 121

    def sample(self, num_samples, device):
        action, log_prob, seq_length = self.sample_action(num_samples=num_samples, device=device)
        smiles = self.char_dict.matrix_to_smiles(action, seq_length - 1)

        return smiles, action, log_prob, seq_length

    def sample_action(self, num_samples, device):
        number_batches = (
            num_samples + self.max_sampling_batch_size - 1
        ) // self.max_sampling_batch_size
        remaining_samples = num_samples

        action = torch.LongTensor(num_samples, self.max_seq_length).to(device)
        log_prob = torch.FloatTensor(num_samples, self.max_seq_length).to(device)
        seq_length = torch.LongTensor(num_samples).to(device)

        batch_start = 0

        for i in range(number_batches):
            batch_size = min(self.max_sampling_batch_size, remaining_samples)
            batch_end = batch_start + batch_size

            action_batch, log_prob_batch, seq_length_batch = self._sample_action_batch(
                batch_size, device
            )
            action[batch_start:batch_end, :] = action_batch
            log_prob[batch_start:batch_end, :] = log_prob_batch
            seq_length[batch_start:batch_end] = seq_length_batch

            batch_start += batch_size
            remaining_samples -= batch_size

        return action, log_prob, seq_length

    def train_on_batch(self, smis, device, weights=1.0):
        actions, _ = smis_to_actions(self.char_dict, smis)
        actions = torch.LongTensor(actions)
        loss = self.train_on_action_batch(actions=actions, device=device, weights=weights)
        return loss

    def train_on_action_batch(self, actions, device, weights=1.0):
        batch_size = actions.size(0)
        batch_seq_length = actions.size(1)

        actions = actions.to(device)

        start_token_vector = self._get_start_token_vector(batch_size, device)
        input_actions = torch.cat([start_token_vector, actions[:, :-1]], dim=1)
        target_actions = actions

        input_actions = input_actions.to(device)
        target_actions = target_actions.to(device)

        output, _ = self.model(input_actions, hidden=None)
        output = output.view(batch_size * batch_seq_length, -1)

        log_probs = torch.log_softmax(output, dim=1)
        log_target_probs = log_probs.gather(dim=1, index=target_actions.reshape(-1, 1)).squeeze(
            dim=1
        )
        log_target_probs = log_target_probs.view(batch_size, batch_seq_length).mean(dim=1)
        loss = -(weights * log_target_probs).mean()

        if self.entropy_factor > 0.0:
            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=1).mean()
            loss -= self.entropy_factor * entropy

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def get_action_log_prob(self, actions, seq_lengths, device):
        num_samples = actions.size(0)
        actions_seq_length = actions.size(1)
        log_probs = torch.FloatTensor(num_samples, actions_seq_length).to(device)

        number_batches = (
            num_samples + self.max_sampling_batch_size - 1
        ) // self.max_sampling_batch_size
        remaining_samples = num_samples
        batch_start = 0
        for i in range(number_batches):
            batch_size = min(self.max_sampling_batch_size, remaining_samples)
            batch_end = batch_start + batch_size
            log_probs[batch_start:batch_end, :] = self._get_action_log_prob_batch(
                actions[batch_start:batch_end, :], seq_lengths[batch_start:batch_end], device
            )
            batch_start += batch_size
            remaining_samples -= batch_size

        return log_probs

    def save(self, save_dir):
        self.model.save(save_dir)

    def _get_action_log_prob_batch(self, actions, seq_lengths, device):
        batch_size = actions.size(0)
        actions_seq_length = actions.size(1)

        start_token_vector = self._get_start_token_vector(batch_size, device)
        input_actions = torch.cat([start_token_vector, actions[:, :-1]], dim=1)
        target_actions = actions

        input_actions = input_actions.to(device)
        target_actions = target_actions.to(device)

        output, _ = self.model(input_actions, hidden=None)
        output = output.view(batch_size * actions_seq_length, -1)
        log_probs = torch.log_softmax(output, dim=1)
        log_target_probs = log_probs.gather(dim=1, index=target_actions.reshape(-1, 1)).squeeze(
            dim=1
        )
        log_target_probs = log_target_probs.view(batch_size, self.max_seq_length)

        mask = torch.arange(actions_seq_length).expand(len(seq_lengths), actions_seq_length) > (
            seq_lengths - 1
        ).unsqueeze(1)
        log_target_probs[mask] = 0.0

        return log_target_probs

    def _sample_action_batch(self, batch_size, device):
        hidden = None
        inp = self._get_start_token_vector(batch_size, device)

        action = torch.zeros((batch_size, self.max_seq_length), dtype=torch.long).to(device)
        log_prob = torch.zeros((batch_size, self.max_seq_length), dtype=torch.float).to(device)
        seq_length = torch.zeros(batch_size, dtype=torch.long).to(device)

        ended = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for t in range(self.max_seq_length):
            output, hidden = self.model(inp, hidden)

            prob = torch.softmax(output, dim=2)
            distribution = Categorical(probs=prob)
            action_t = distribution.sample()
            log_prob_t = distribution.log_prob(action_t)
            inp = action_t

            action[~ended, t] = action_t.squeeze(dim=1)[~ended]
            log_prob[~ended, t] = log_prob_t.squeeze(dim=1)[~ended]

            seq_length += (~ended).long()
            ended = ended | (action_t.squeeze(dim=1) == self.char_dict.end_idx).bool()

            if ended.all():
                break

        return action, log_prob, seq_length

    def _get_start_token_vector(self, batch_size, device):
        return torch.LongTensor(batch_size, 1).fill_(self.char_dict.begin_idx).to(device)
