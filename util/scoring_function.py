
import networkx as nx
import os, sys

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import Descriptors
from util.SA_Score import sascorer
from vina import Vina
from openbabel import pybel as pyb







def _penalized_logp_cyclebasis(mol):
    log_p = Descriptors.MolLogP(mol)
    sa_score = sascorer.calculateScore(mol)
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sa_score - cycle_score




class penalized_score:
    def score(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return _penalized_logp_cyclebasis(mol)
    


def docking(lig,name,tar):
    v = Vina(sf_name='vina')
    v.set_receptor('./data/'+tar+'.pdbqt')
    v.set_ligand_from_file(lig)
    v.compute_vina_maps(center=[15.190, 53.903, 16.917], box_size=[20, 20, 20])
    v.dock(exhaustiveness=32, n_poses=20)
    v.write_poses('./test/'+name+"_docked.pdbqt", n_poses=1, overwrite=True)



class docking_score:
    def score(self, smiles,target):
        mymol = pybel.readstring("smi", smiles)
        mymol.make3D()
        mymol.write(format='pdbqt', filename='./log/test.pdbqt',overwrite=True)
        return docking('./log/test.pdbqt','./log/test.pdbqt',target)[0]

