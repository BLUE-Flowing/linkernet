import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.transforms import BaseTransform
import sys
import os
from easydict import EasyDict
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit import Geometry
from datasets.linker_data import FragLinkerData, torchify_dict

sys.path.append('.')
belong_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(belong_dir)

from utils import so3
from utils.geometry import local_to_global, global_to_local, rotation_matrix_from_vectors, find_axes
from torch_scatter import scatter_add

### add
ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
# required by model:
# x_noisy, c_noisy, atom_feat, edge_index, edge_feat,
# R1_noisy, R2_noisy, R1, R2, eps_p1, eps_p2, x_0, c_0, t, fragment_mask, batch


def modify_frags_conformer(frags_local_pos, frag_idx_mask, v_frags, p_frags):
    R_frags = so3.so3vec_to_rotation(v_frags)
    x_frags = torch.zeros_like(frags_local_pos)
    for i in range(2):
        noisy_pos = local_to_global(R_frags[i], p_frags[i],
                                    frags_local_pos[frag_idx_mask == i + 1])
        x_frags[frag_idx_mask == i + 1] = noisy_pos
    return x_frags


def dataset_info(dataset):  # qm9, zinc, cep
    if dataset == 'qm9':
        return {'atom_types': ["H", "C", "N", "O", "F"],
                'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
                }
    elif dataset == 'zinc' or dataset == 'protac':
        return {'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 
                               'I1(0)', 'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 
                               'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)'],
                'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4,
                                    13: 6, 14: 3},
                'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                                   10: 'O', 11: 'S', 12: 'S', 13: 'S'},
                'bucket_sizes': np.array(
                    [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58, 84])
                }

    elif dataset == "cep":
        return {'atom_types': ["C", "S", "N", "O", "Se", "Si"],
                'maximum_valence': {0: 4, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4},
                'number_to_atom': {0: "C", 1: "S", 2: "N", 3: "O", 4: "Se", 5: "Si"},
                'bucket_sizes': np.array([25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 43, 46])
                }
    else:
        print("the datasets in use are qm9|zinc|cep")
        exit(1)


class FeaturizeAtom(BaseTransform):

    def __init__(self, dataset_name, known_anchor=False,
                 add_atom_type=True, add_atom_feat=True):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info(dataset_name)
        self.known_anchor = known_anchor
        self.add_atom_type = add_atom_type
        self.add_atom_feat = add_atom_feat

    @property
    def num_classes(self):
        return len(self.dataset_info['atom_types'])

    @property
    def feature_dim(self):
        n_feat_dim = 2
        if self.known_anchor:
            n_feat_dim += 2
        return n_feat_dim

    def get_index(self, atom_num, valence, charge):
        if self.dataset_name in ['zinc', 'protac']:
            pt = Chem.GetPeriodicTable()
            atom_str = "%s%i(%i)" % (pt.GetElementSymbol(atom_num), valence, charge)
            return self.dataset_info['atom_types'].index(atom_str)
        else:
            raise ValueError

    def get_element_from_index(self, index):
        pt = Chem.GetPeriodicTable()
        symb = self.dataset_info['number_to_atom'][index]
        return pt.GetAtomicNumber(symb)

    def __call__(self, data):
        if self.add_atom_type:
            x = [self.get_index(int(e), int(v), int(c)) for e, v, c in zip(data.element, data.valence, data.charge)]
            data.atom_type = torch.tensor(x)
        if self.add_atom_feat:
            # fragment / linker indicator, independent with atom types
            linker_flag = F.one_hot((data.fragment_mask == 0).long(), 2)
            all_feats = [linker_flag]
            # fragment anchor flag
            if self.known_anchor:
                anchor_flag = F.one_hot((data.anchor_mask == 1).long(), 2)
                all_feats.append(anchor_flag)
            data.atom_feat = torch.cat(all_feats, -1)
        return data


class BuildCompleteGraph(BaseTransform):

    def __init__(self, known_linker_bond=False, known_cand_anchors=False):
        super().__init__()
        self.known_linker_bond = known_linker_bond
        self.known_cand_anchors = known_cand_anchors

    @property
    def num_bond_classes(self):
        return 5

    @property
    def bond_feature_dim(self):
        return 4

    @staticmethod
    def _get_interleave_edge_index(edge_index):
        edge_index_sym = torch.stack([edge_index[1], edge_index[0]])
        e = torch.zeros_like(torch.cat([edge_index, edge_index_sym], dim=-1))
        e[:, ::2] = edge_index
        e[:, 1::2] = edge_index_sym
        return e

    def _build_interleave_fc(self, n1_atoms, n2_atoms):
        eij = torch.triu_indices(n1_atoms, n2_atoms, offset=1)
        e = self._get_interleave_edge_index(eij)
        return e

    def __call__(self, data):
        # fully connected graph
        num_nodes = len(data.pos)
        fc_edge_index = self._build_interleave_fc(num_nodes, num_nodes)
        data.edge_index = fc_edge_index

        # (ll, lf, fl, ff) indicator
        src, dst = data.edge_index
        num_edges = len(fc_edge_index[0])
        edge_type = torch.zeros(num_edges).long()
        l_ind_src = data.fragment_mask[src] == 0    # not masked src index
        l_ind_dst = data.fragment_mask[dst] == 0    # not masked dst index
        edge_type[l_ind_src & l_ind_dst] = 0
        edge_type[l_ind_src & ~l_ind_dst] = 1
        edge_type[~l_ind_src & l_ind_dst] = 2
        edge_type[~l_ind_src & ~l_ind_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4) ## test record the edge type(if no linker left, then edge_type[i] == [0, 0, 0, 1])
        data.edge_feat = edge_type

        # bond type 0: none 1: singe 2: double 3: triple 4: aromatic
        bond_type = torch.zeros(num_edges).long()

        id_fc_edge = fc_edge_index[0] * num_nodes + fc_edge_index[1] ## all possible edge index
        id_frag_bond = data.bond_index[0] * num_nodes + data.bond_index[1]
        idx_edge = torch.tensor([torch.nonzero(id_fc_edge == id_).squeeze() for id_ in id_frag_bond])
        bond_type[idx_edge] = data.bond_type
        # data.edge_type = F.one_hot(bond_type, num_classes=5)
        data.edge_type = bond_type
        if self.known_linker_bond:
            data.linker_bond_mask = (data.fragment_mask[src] == 0) ^ (data.fragment_mask[dst] == 0)
        elif self.known_cand_anchors:
            ll_bond = (data.fragment_mask[src] == 0) & (data.fragment_mask[dst] == 0)
            fl_bond = (data.cand_anchors_mask[src] == 1) & (data.fragment_mask[dst] == 0)
            lf_bond = (data.cand_anchors_mask[dst] == 1) & (data.fragment_mask[src] == 0)
            data.linker_bond_mask = ll_bond | fl_bond | lf_bond
        else:
            data.linker_bond_mask = (data.fragment_mask[src] == 0) | (data.fragment_mask[dst] == 0)

        data.inner_edge_mask = (data.fragment_mask[src] == data.fragment_mask[dst])
        return data


class SelectCandAnchors(BaseTransform):

    def __init__(self, mode='exact', k=2):
        super().__init__()
        self.mode = mode
        assert mode in ['exact', 'k-hop']
        self.k = k

    @staticmethod
    def bfs(nbh_list, node, k=2, valid_list=[]):
        visited = [node]
        queue = [node]
        level = [0]
        bfs_perm = []

        while len(queue) > 0:
            m = queue.pop(0)
            l = level.pop(0)
            if l > k:
                break
            bfs_perm.append(m)

            for neighbour in nbh_list[m]:
                if neighbour not in visited and neighbour in valid_list:
                    visited.append(neighbour)
                    queue.append(neighbour)
                    level.append(l + 1)
        return bfs_perm

    def __call__(self, data):
        # link_indices = (data.linker_mask == 1).nonzero()[:, 0].tolist()
        # frag_indices = (data.linker_mask == 0).nonzero()[:, 0].tolist()
        # anchor_indices = [j for i, j in zip(*data.bond_index.tolist()) if i in link_indices and j in frag_indices]
        # data.anchor_indices = anchor_indices
        cand_anchors_mask = torch.zeros_like(data.fragment_mask).bool()
        if self.mode == 'exact':
            cand_anchors_mask[data.anchor_indices] = True
            data.cand_anchors_mask = cand_anchors_mask

        elif self.mode == 'k-hop':
            # data.nbh_list = {i.item(): [j.item() for k, j in enumerate(data.bond_index[1])
            #                             if data.bond_index[0, k].item() == i] for i in data.bond_index[0]}
            # all_cand = []
            #for anchor in data.anchor_indices:
            #    a_frag_id = data.fragment_mask[anchor]
            #    a_valid_list = (data.fragment_mask == a_frag_id).nonzero(as_tuple=True)[0].tolist()
            #    a_cand = self.bfs(data.nbh_list, anchor, k=self.k, valid_list=a_valid_list)
            #    #a_cand = [a for a in a_cand if data.frag_mol.GetAtomWithIdx(a).GetTotalNumHs() > 0]
            #    a_cand = [anchor]
            #    cand_anchors_mask[a_cand] = True
                # all_cand.append(a_cand)
            #data.cand_anchors_mask = cand_anchors_mask
            pass
        else:
            raise ValueError(self.mode)
        return data


class StackFragLocalPos(BaseTransform):
    def __init__(self, max_num_atoms=30):
        super().__init__()
        self.max_num_atoms = max_num_atoms

    def __call__(self, data):
        frag_idx_mask = data.fragment_mask[data.fragment_mask > 0]
        f1_pos = data.frags_local_pos[frag_idx_mask == 1]
        f2_pos = data.frags_local_pos[frag_idx_mask == 2]
        assert len(f1_pos) <= self.max_num_atoms
        assert len(f2_pos) <= self.max_num_atoms
        # todo: use F.pad
        f1_fill_pos = torch.cat([f1_pos, torch.zeros(self.max_num_atoms - len(f1_pos), 3)], dim=0)
        f1_mask = torch.cat([torch.ones(len(f1_pos)), torch.zeros(self.max_num_atoms - len(f1_pos))], dim=0)
        f2_fill_pos = torch.cat([f2_pos, torch.zeros(self.max_num_atoms - len(f2_pos), 3)], dim=0)
        f2_mask = torch.cat([torch.ones(len(f2_pos)), torch.zeros(self.max_num_atoms - len(f2_pos))], dim=0)
        data.frags_local_pos_filled = torch.stack([f1_fill_pos, f2_fill_pos], dim=0)
        data.frags_local_pos_mask = torch.stack([f1_mask, f2_mask], dim=0).bool()
        return data


class RelativeGeometry(BaseTransform):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def __call__(self, data):
        if self.mode == 'relative_pos_and_rot':
            # randomly take first / second fragment as the reference
            idx = torch.randint(0, 2, [1])[0]
            pos = (data.pos - data.frags_t[idx]) @ data.frags_R[idx]
            frags_R = data.frags_R[idx].T @ data.frags_R
            frags_t = (data.frags_t - data.frags_t[idx]) @ data.frags_R[idx]
            # frags_d doesn't change
            data.frags_rel_mask = torch.tensor([True, True])
            data.frags_rel_mask[idx] = False  # the reference fragment will not be added noise later
            data.frags_atom_rel_mask = data.fragment_mask == (2 - idx)

        elif self.mode == 'two_pos_and_rot':
            # still guarantee the center of two fragments' centers is the origin
            rand_rot = get_random_rot()
            pos = data.pos @ rand_rot
            frags_R = rand_rot.T @ data.frags_R
            frags_t = data.frags_t @ rand_rot
            data.frags_rel_mask = torch.tensor([True, True])

        elif self.mode == 'distance_and_two_rot_aug':
            # only the first row of frags_R unchanged
            rand_rot = get_random_rot()
            tmp_pos = data.pos @ rand_rot
            tmp_frags_R = rand_rot.T @ data.frags_R
            tmp_frags_t = data.frags_t @ rand_rot

            rot = rotation_matrix_from_vectors(tmp_frags_t[1] - tmp_frags_t[0], torch.tensor([1., 0., 0.]))
            tr = -rot @ ((tmp_frags_t[0] + tmp_frags_t[1]) / 2)
            pos = tmp_pos @ rot.T + tr
            frags_R = rot @ tmp_frags_R
            frags_t = tmp_frags_t @ rot.T + tr
            data.frags_rel_mask = torch.tensor([True, True])

        elif self.mode == 'distance_and_two_rot':
            # unchanged
            frags_R = data.frags_R
            frags_t = data.frags_t
            pos = data.pos
            data.frags_rel_mask = torch.tensor([True, True])

        else:
            raise ValueError(self.mode)

        data.frags_R = frags_R
        data.frags_t = frags_t
        data.pos = pos
        # print('frags_R: ', data.frags_R,  'frags_t: ', frags_t)
        return data


def get_random_rot():
    M = np.random.randn(3, 3)
    Q, __ = np.linalg.qr(M)
    rand_rot = torch.from_numpy(Q.astype(np.float32))
    return rand_rot


class ReplaceLocalFrame(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        frags_R1 = get_random_rot()
        frags_R2 = get_random_rot()
        f1_local_pos = global_to_local(frags_R1, data.frags_t[0], data.pos[data.fragment_mask == 1])
        f2_local_pos = global_to_local(frags_R2, data.frags_t[1], data.pos[data.fragment_mask == 2])
        data.frags_local_pos = torch.cat([f1_local_pos, f2_local_pos], dim=0)
        return data
    
    
class GenerateLinkerDataset(object):
    def __init__(self, fragment_1_path, fragment_2_path=None):
        super().__init__()
        self.fragment_1_path = fragment_1_path
        self.fragment_2_path = fragment_2_path
        if self.fragment_2_path is not None:    ## mode: given two-frags 
            [data, atom_indices_f1, atom_indices_f2] = self._combine_fragemnt()
            self.data = data
            self.atom_indices_f1 = atom_indices_f1
            self.atom_indices_f2 = atom_indices_f2
        elif self.fragment_2_path is None:      ## mode: given whole molecule
            ### for this condition, cannot support kekulize bond right now. 
            data = self._parse_rdmol(Chem.MolFromMolFile(self.fragment_1_path))
            new_data = self._process_single_mol(data)
            self.data = new_data
            self.atom_indices_f1 = new_data['atom_indices_f1']
            self.atom_indices_f2 = new_data['atom_indices_f2']
            
    def _process_single_mol(self, data):
        nbh_list = {i.item(): [j.item() for k, j in enumerate(data['bond_index'][1])
                                    if data['bond_index'][0, k].item() == i] for i in data['bond_index'][0]}
        atom_nbh_valence = self._count_neighbors(edge_index=data['bond_index'], valence=data['bond_type'])
        left_select = data['valence'] - atom_nbh_valence
        atom_indices = np.arange(len(data['element'])).tolist()
        select_candidate = left_select[atom_indices] != 0
        
        new_data = {}
        new_data['anchor_indices'] = []
        
        while len(new_data['anchor_indices']) < 2:
            if select_candidate.size > 0:
                linker_mask_first_point = np.random.choice((np.where(select_candidate))[0])
            else:
                linker_mask_first_point = np.random.choice(np.arange(len(data['element'])))
        
            ### linker mask index bfs generate
            linker_bfs_idx = bfs(node=linker_mask_first_point, k=1, nbh_list=nbh_list, valid_list=atom_indices)
            frag_idx = [i for i in atom_indices if atom_indices[i] not in linker_bfs_idx]
            new_data['linker_nums'] = len(linker_bfs_idx)

            ## unmasked bond info
            new_bond_index = [[data['bond_index'].tolist()[0][i], data['bond_index'].tolist()[1][i]] 
                                  for i in range(len(data['bond_index'].tolist()[0])) 
                                  if data['bond_index'].tolist()[0][i] not in linker_bfs_idx and data['bond_index'].tolist()[1][i] not in linker_bfs_idx]
            select_bond_bool = [
                (bond_index_1 not in linker_bfs_idx) and (bond_index_2 not in linker_bfs_idx)
                for bond_index_1, bond_index_2 in zip(*data['bond_index'])
            ]
            ## change-to-new_index 
            frag_order = {i : key for key, i in enumerate(frag_idx)}
            
            # bond info
            bond_index = data['bond_index'][:, np.where(select_bond_bool)[0].tolist()]
            bond_type = data['bond_type'][np.where(select_bond_bool)[0].tolist()]
            
            # f1 and f2 info   |   frag_idx is ordered
            ori_frag_nbh_list = {key: [j for j in value[1] if j not in linker_bfs_idx] for key, value in enumerate(nbh_list.items()) if key not in linker_bfs_idx}
            
            anchors = [i for i in frag_idx if any(value in nbh_list[i] for value in linker_bfs_idx)]
            have_points = False
            if len(anchors) == 2: ### only two anchors
                atom_indices_f1 = bfs(nbh_list=ori_frag_nbh_list, k=len(frag_idx), valid_list=frag_idx, node=anchors[0])
                atom_indices_f2 = bfs(nbh_list=ori_frag_nbh_list, k=len(frag_idx), valid_list=frag_idx, node=anchors[1])
                if len(atom_indices_f1) + len(atom_indices_f2) != len(frag_idx):
                    print(f'Select atom mistake. Try random choice again.')
                elif len(atom_indices_f1) + len(atom_indices_f2) == len(frag_idx) and len(atom_indices_f1) != 0 and len(atom_indices_f2) != 0:
                    for _, value in ori_frag_nbh_list.items():
                        if len(value) == 0:
                            have_points = True
                    if not have_points:
                        get_atom_indices_f1, get_atom_indices_f2 = np.array([i for i in atom_indices_f1]), np.array([i for i in atom_indices_f2])
                        atom_indices = [*sorted(get_atom_indices_f1.tolist()), *sorted(get_atom_indices_f2.tolist())]
                        new_order = {i: [key for key, value in enumerate(atom_indices) if value == i] for i in frag_idx}
                        new_order = {i : value[0] for i, value in new_order.items()}
                        new_order = dict(sorted(new_order.items(), key=lambda item: item[1]))
                        ## get new_ordered fragment index and info
                        new_data['atom_indices_f1'] = [new_order[i] for i in sorted(get_atom_indices_f1.tolist())]
                        new_data['atom_indices_f2'] = [new_order[i] for i in sorted(get_atom_indices_f2.tolist())]
                        new_data['element'] = np.array([data['element'].tolist()[i] for i, _ in new_order.items()])
                        new_data['pos'] = np.array([data['pos'].tolist()[i] for i, _ in new_order.items()], dtype=np.float32)
                        new_data['valence'] = np.array([data['valence'].tolist()[i] for i, _ in new_order.items()])
                        new_data['charge'] = np.array([data['charge'].tolist()[i] for i, _ in new_order.items()])
                        new_data['hybridization'] = [data['hybridization'][i] for i, _ in new_order.items()]
                        new_data['bond_index'] = np.array([[new_order[i] for i in bond_index[j]] for j in range(bond_index.shape[0])])
                        new_data['bond_type'] = bond_type
                        new_data['nbh_list'] = {new_order[key] : [new_order[j] for j in value] for key, value in ori_frag_nbh_list.items()}
                        new_data['fragment_mask'] = torch.cat([
                                torch.ones(len(new_data['atom_indices_f1']), dtype=torch.int32), torch.ones(len(new_data['atom_indices_f2']), dtype=torch.int32) *2
                            ], dim=0)
                        new_data['linker_mask'] = torch.zeros_like(new_data['fragment_mask'], dtype=torch.bool)
                        new_data['anchor_indices'] = [new_order[anchors[0]], new_order[anchors[1]]]
                        cand_anchors_mask = np.zeros((new_data['fragment_mask'].numpy().size), dtype=bool)
                        cand_anchors_mask[new_order[anchors[0]]] ,cand_anchors_mask[new_order[anchors[1]]] = True, True
                        new_data['cand_anchors_mask'] = cand_anchors_mask
                    else:
                        print(f'Select atom mistake. Try random choice again.')
            
            elif len(anchors) == 3: ## right now just consider 3 anchor condition
                for i in range(len(anchors)):
                    j = (i+1) % len(anchors)
                    k = (i+2) % len(anchors)
                    atom_indices_f1 = [*bfs(nbh_list=ori_frag_nbh_list, k=len(frag_idx), valid_list=frag_idx, node=anchors[i]), 
                                       *bfs(nbh_list=ori_frag_nbh_list, k=len(frag_idx), valid_list=frag_idx, node=anchors[j])]
                    atom_indices_f2 = bfs(nbh_list=ori_frag_nbh_list, k=len(frag_idx), valid_list=frag_idx, node=anchors[k])
                    if len(atom_indices_f1) + len(atom_indices_f2) == len(frag_idx) and len(atom_indices_f1) > 1 and len(atom_indices_f2) > 1:
                        for _, value in ori_frag_nbh_list.items():
                            if len(value) == 0:
                                have_points = True
                        if not have_points:     
                            get_atom_indices_f1, get_atom_indices_f2 = np.array([i for i in atom_indices_f1]), np.array([i for i in atom_indices_f2])
                            atom_indices = [*sorted(get_atom_indices_f1.tolist()), *sorted(get_atom_indices_f2.tolist())]
                            new_order = {i: [key for key, value in enumerate(atom_indices) if value == i] for i in frag_idx}
                            new_order = {i : value[0] for i, value in new_order.items()}
                            ## get new_ordered fragment index and info
                            new_data['atom_indices_f1'] = [new_order[i] for i in sorted(get_atom_indices_f1.tolist())]
                            new_data['atom_indices_f2'] = [new_order[i] for i in sorted(get_atom_indices_f2.tolist())]
                            new_data['element'] = np.array([data['element'].tolist()[i] for i, _ in new_order.items()])
                            new_data['pos'] = np.array([data['pos'].tolist()[i] for i, _ in new_order.items()], dtype=np.float32)
                            new_data['valence'] = np.array([data['valence'].tolist()[i] for i, _ in new_order.items()])
                            new_data['charge'] = np.array([data['charge'].tolist()[i] for i, _ in new_order.items()])
                            new_data['hybridization'] = [data['hybridization'][i] for i, _ in new_order.items()]
                            new_data['bond_index'] = np.array([[new_order[i] for i in bond_index[j]] for j in range(bond_index.shape[0])])
                            new_data['bond_type'] = bond_type
                            new_data['nbh_list'] = {new_order[key] : [new_order[j] for j in value] for key, value in ori_frag_nbh_list.items()}
                            new_data['fragment_mask'] = torch.cat([
                                    torch.ones(len(new_data['atom_indices_f1']), dtype=torch.int32), torch.ones(len(new_data['atom_indices_f2']), dtype=torch.int32) *2
                                ], dim=0)
                            new_data['linker_mask'] = torch.zeros_like(new_data['fragment_mask'], dtype=torch.bool)
                            f1_anchor = np.random.choice([new_order[anchors[i]], new_order[anchors[j]]])
                            f2_anchor = new_order[anchors[k]]
                            new_data['anchor_indices'] = [f1_anchor, f2_anchor]
                        else:
                            print(f'Select atom mistake. Try random choice again.')
            elif len(anchors) < 2 or len(anchors) > 3:
                print(f'Select atom mistake. Try random choice again.')
                                     
            #atom_indices_f1 = np.unique(np.array([i for i in frag_idx for j in linker_bfs_idx if i < j]))
            #atom_indices_f2 = np.unique(np.array([i for i in frag_idx if i not in atom_indices_f1]))
            ### get_anchor_idx
            #anchor_flag = False
            #if len(ordered_atom_indices_f1.tolist()) != 0 and len(ordered_atom_indices_f2.tolist()) != 0:
            #    f1_candidate = [frag_order[i] for i in atom_indices_f1 if select_candidate[i] == True and any(nbh_list[i]) in linker_bfs_idx]
            #    f2_candidate = [frag_order[j] for j in atom_indices_f2 if select_candidate[j] == True and any(nbh_list[j]) in linker_bfs_idx]
            #    if len(f1_candidate) != 0 and len(f2_candidate) != 0: 
            #        for _, value in ori_frag_nbh_list.items():
            #            if len(value) == 0: # some atom is not aligned to the fragments
            #                anchor_flag = False
            #                break
            #        if anchor_flag: 
            #            f1_anchor = np.random.choice(np.array(f1_candidate))
            #            f2_anchor = np.random.choice(np.array(f2_candidate))
            #            new_data['anchor_indices'] = [f1_anchor, f2_anchor]
            #    else:
            #        ## reselect a linker begin atom
            #        print(f'Select atom mistake. Try random choice again.')
            #else:
            #    ## reselect a linker begin atom
            #    print(f'Select atom mistake. Try random choice again.')
        
        return new_data
        
    def _combine_fragemnt(self):
        fragment_1 = Chem.MolFromMolFile(self.fragment_1_path)
        fragment_2 = Chem.MolFromMolFile(self.fragment_2_path)
        fragment_1_data = self._parse_rdmol(fragment_1)
        fragment_2_data = self._parse_rdmol(fragment_2)
        f_1_num = len(fragment_1_data['element'])
        f_2_num = len(fragment_2_data['element'])
        
        ### combine
        data = {}
        data['element'] = np.concatenate((fragment_1_data['element'], fragment_2_data['element']), axis=0)
        data['bond_index'] = np.concatenate((fragment_1_data['bond_index'], fragment_2_data['bond_index'] + f_1_num), axis=1)
        data['bond_type'] = np.concatenate((fragment_1_data['bond_type'], fragment_2_data['bond_type']), axis=0)
        data['pos'] = np.concatenate((fragment_1_data['pos'], fragment_2_data['pos']), axis=0)
        data['valence'] = np.concatenate((fragment_1_data['valence'], fragment_2_data['valence']), axis=0)
        data['charge'] = np.concatenate((fragment_1_data['charge'], fragment_2_data['charge']), axis=0)
        data['hybridization'] = [*fragment_1_data['hybridization'], *fragment_2_data['hybridization']]
        
        ## get no-linker related mask info
        data['fragment_mask'] = torch.cat([
            torch.ones(f_1_num, dtype=torch.int32), torch.ones(f_2_num, dtype=torch.int32) *2
        ], dim=0)
        data['linker_mask'] = torch.zeros(f_1_num + f_2_num, dtype=torch.bool)
          
        ### nbh_list
        nbh_list = {i.item(): [j.item() for k, j in enumerate(data['bond_index'][1])
                                    if data['bond_index'][0, k].item() == i] for i in data['bond_index'][0]}
        data['nbh_list'] = nbh_list
        
        ### add linker_mask
        atom_indices_f1 = np.arange(f_1_num).tolist()
        atom_indices_f2 = (np.arange(f_2_num) + f_1_num).tolist()
        data['atom_indices_f1'] = atom_indices_f1 
        data['atom_indices_f2'] = atom_indices_f2
        
        # TEST anchor find randomly
        ### find possible anchor
        atom_nbh_valence = self._count_neighbors(edge_index=data['bond_index'], valence=data['bond_type'])
        data['atom_nbh_valence'] = atom_nbh_valence
        
        
        
        
        return [data, atom_indices_f1, atom_indices_f2]
     
    @staticmethod
    def _count_neighbors(edge_index, symmetry=True, valence=None):
        assert symmetry == True, 'Only support symmetrical edges.'              
        valence = valence.reshape(edge_index.shape[1])
        atom_index = np.unique(edge_index[0]).tolist()
        edge_type = valence
        atom_valence = []
        for idx in atom_index:
            num = 0
            for i, j in enumerate(edge_index[0].tolist()):
                if idx == j: # index == i
                    if edge_type[i] == 4:
                        num = num + 1.5
                    else:
                        num = num + edge_type[i]
            num = int(num)
            atom_valence.append(num)
        return atom_valence
    
    @staticmethod
    def _parse_rdmol(rdmol, kekulize=True):   
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

        rd_num_atoms = rdmol.GetNumAtoms()
        feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
        for feat in factory.GetFeaturesForMol(rdmol):
            feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

        # Get hybridization in the order of atom idx.
        hybridization = []
        for atom in rdmol.GetAtoms():
            hybr = str(atom.GetHybridization())
            idx = atom.GetIdx()
            hybridization.append((idx, hybr))
        hybridization = sorted(hybridization)
        hybridization = [v[1] for v in hybridization]

        ptable = Chem.GetPeriodicTable()

        pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
        element, valence, charge = [], [], []
        accum_pos = 0
        accum_mass = 0
        for atom_idx in range(rd_num_atoms):
            atom = rdmol.GetAtomWithIdx(atom_idx)
            atom_num = atom.GetAtomicNum()
            element.append(atom_num)
            valence.append(atom.GetTotalValence())
            charge.append(atom.GetFormalCharge())
            atom_weight = ptable.GetAtomicWeight(atom_num)
            accum_pos += pos[atom_idx] * atom_weight
            accum_mass += atom_weight
        center_of_mass = accum_pos / accum_mass
        element = np.array(element, dtype=int)
        valence = np.array(valence, dtype=int)
        charge = np.array(charge, dtype=int)

        # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
        row, col, edge_type = [], [], []
        for bond in rdmol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            if kekulize:
                edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]
            elif not kekulize:
                if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    if bond.GetBeginAtom().GetAtomicNum() == bond.GetEndAtom().GetAtomicNum():
                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    else:
                        bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
                    bond.SetIsAromatic(False)
                edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

        edge_index = np.array([row, col], dtype=np.int64)
        edge_type = np.array(edge_type, dtype=int)

        perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]

        data = {
            'rdmol': rdmol,
            'element': element,
            'pos': pos,
            'bond_index': edge_index,
            'bond_type': edge_type,
            'center_of_mass': center_of_mass,
            'atom_feature': feat_mat,
            'hybridization': hybridization,
            'valence': valence,
            'charge': charge
        }
        return data
    
    @property
    def _get_data(self):
        return self.data
    
    @property
    def _get_indices_1(self):
        return self.atom_indices_f1
    
    @property
    def _get_indices_2(self):
        return self.atom_indices_f2
    
    def _parse_sdf_file(self, path):
        fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
 
        rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=False)))
        if rdmol is None:
            raise ValueError('Invalid SDF file: %s' % path)
    
        rd_num_atoms = rdmol.GetNumAtoms()
        feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=int)
        # create a one-hot index dict
        for feat in factory.GetFeaturesForMol(rdmol):
            feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

        with open(path, 'r') as f:
            sdf = f.read()

        sdf = sdf.splitlines()
        num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])
        assert num_atoms == rd_num_atoms

        hybridization = []
        for atom in rdmol.GetAtoms():
            hybr = str(atom.GetHybridization())
            idx = atom.GetIdx()
            hybridization.append((idx, hybr))
        hybridization = sorted(hybridization)
        hybridization = [v[1] for v in hybridization]
        
        ptable = Chem.GetPeriodicTable()

        element, pos = [], []
        valance, charge = [], []
        accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        accum_mass = 0.0
        for atom_line in map(lambda x:x.split(), sdf[4:4+num_atoms]):
            x, y, z = map(float, atom_line[:3])
            symb = atom_line[3]
            atomic_number = ptable.GetAtomicNumber(symb.capitalize())
            element.append(atomic_number)
            pos.append([x, y, z])
            atomic_weight = ptable.GetAtomicWeight(atomic_number)
            accum_pos += np.array([x, y, z]) * atomic_weight
            accum_mass += atomic_weight

        center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)

        element = np.array(element, dtype=int)
        pos = np.array(pos, dtype=np.float32)
        BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
        bond_type_map = {
            1: BOND_TYPES[BondType.SINGLE],
            2: BOND_TYPES[BondType.DOUBLE],
            3: BOND_TYPES[BondType.TRIPLE],
            4: BOND_TYPES[BondType.AROMATIC],
        }
    
        row, col, edge_type = [], [], []
        for bond_line in sdf[4+num_atoms:4+num_atoms+num_bonds]:
            start, end = int(bond_line[0:3])-1, int(bond_line[3:6])-1
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]

        # the edge index is double-wards
        edge_index = np.array([row, col], dtype=np.int64) 
        edge_type = np.array(edge_type, dtype=int)

        # get new ordered edge index
        perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        
        # get atom_valance
        atom_valance = self._count_neighbors(
            edge_index, 
            symmetry=True, 
            valence=edge_type,
            num_nodes=len(element),
        )
        #test data 还需要电量
        data = {
            'element': element,
            'pos': pos,
            'bond_index': edge_index,
            'bond_type': edge_type,
            'center_of_mass': center_of_mass,
            'atom_feature': feat_mat,
            'valance': atom_valance,
        }
        return data

class GetRandomAnchor(object):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def _get_anchor(self):
        ### get basic data
        atom_nbh_valence = self.data['atom_nbh_valence']
        atom_indices_f1 = self.data['atom_indices_f1']
        atom_indices_f2 = self.data['atom_indices_f2']
        nbh_list = self.data['nbh_list']
        
        ### choose random anchor
        left_select = self.data['valence'] - atom_nbh_valence
        select_candidate_1 = left_select[atom_indices_f1] != 0
        select_candidate_2 = left_select[atom_indices_f2] != 0
        true_indices_1 = np.where(select_candidate_1 == True)[0]
        true_indices_2 = np.where(select_candidate_2 == True)[0] + len(atom_indices_f1)
        if true_indices_1.size > 0:
            random_index_1 = np.random.choice(true_indices_1)
        else:
            random_index_1 = np.random.choice(np.arange(len(atom_indices_f1)))
        if true_indices_2.size > 0:
            random_index_2 = np.random.choice(true_indices_2)
        else:
            random_index_2 = np.random.choice(np.arange(len(atom_indices_f1)+len(atom_indices_f1))[len(atom_indices_f1):])  
        self.data['anchor_indices'] = [random_index_1, random_index_2]
        print(f'choose anchor: {random_index_1} , {random_index_2}')
        ### bfs
        bfs_1 = bfs(nbh_list=nbh_list, node=random_index_1, valid_list=atom_indices_f1)
        bfs_2 = bfs(nbh_list=nbh_list, node=random_index_2, valid_list=atom_indices_f2)
        cand_1, cand_2 = [], []
        for i in bfs_1:
            if i in true_indices_1:
                cand_1.append(i)
        for j in bfs_2:
            if j in true_indices_2:
                cand_2.append(j)
        cand_anchors_mask = np.zeros((self.data['fragment_mask'].numpy().size), dtype=bool)
        cand_anchors_mask[cand_1] = True
        cand_anchors_mask[cand_2] = True
        self.data['cand_anchors_mask'] = cand_anchors_mask
    
    @property
    def get_data(self):
        return self.data
        
class RandomFragLinkerDataset(object):
    def __init__(self, data, mode=None):
        super().__init__()
        self.data = data
        self.mode = mode
        self.atom_indices_f1 = data['atom_indices_f1']
        self.atom_indices_f2 = data['atom_indices_f2']
        self.processed_data = []
        
    def _rotation(self):
        frags_t = []
        f1_pos = self.data['pos'][self.atom_indices_f1]
        frags_t.append(f1_pos.mean(-2)) 
        f2_pos = self.data['pos'][self.atom_indices_f2]
        frags_t.append(f2_pos.mean(-2))
        rot = rotation_matrix_from_vectors(
            torch.from_numpy(frags_t[1] - frags_t[0]), torch.tensor([1., 0., 0.])).numpy()   
        tr = -rot @ ((frags_t[0] + frags_t[1]) / 2)
        
        if self.mode == 'distance_and_two_rot':
            self.data['pos'] = self.data['pos']
        else:
            self.data['pos'] = (rot @ self.data['pos'].T).T + tr     
        
        frags_axes = []
        frags_t = []
        for ind_key in ['atom_indices_f1', 'atom_indices_f2']:
            if ind_key == 'atom_indices_f1':
                f_pos = self.data['pos'][self.atom_indices_f1]
                f_charge = np.array(
                    [self.data['element'][i] for i in self.atom_indices_f1], dtype=np.float32)
                axes = find_axes(f_pos, f_charge)
                frags_axes.append(axes)
                frags_t.append(f_pos.mean(-2))
            if ind_key == 'atom_indices_f2':
                f_pos = self.data['pos'][self.atom_indices_f2]
                f_charge = np.array(
                    [self.data['element'][i] for i in self.atom_indices_f2], dtype=np.float32)
                axes = find_axes(f_pos, f_charge)
                frags_axes.append(axes)
                frags_t.append(f_pos.mean(-2))

        self.data['frags_R'] = np.stack(frags_axes)
        self.data['frags_t'] = np.stack(frags_t)
        
        data = FragLinkerData(**torchify_dict(self.data))
        f1_local_pos = global_to_local(data.frags_R[0], data.frags_t[0], data.pos[data.fragment_mask == 1])
        f2_local_pos = global_to_local(data.frags_R[1], data.frags_t[1], data.pos[data.fragment_mask == 2])

        data.frags_local_pos = torch.cat([f1_local_pos.squeeze(0), f2_local_pos.squeeze(0)])
        data.frags_d = data.frags_t[1][0] - data.frags_t[0][0]

        anchor_mask = torch.zeros_like(data.fragment_mask)
        anchor_mask[data.anchor_indices] = 1
        data.anchor_mask = anchor_mask
        
        # add frag_mol and link_mol
        frag_mol = reconstruct_from_generated_with_edges(data)
        link_mol = None
        data.frag_mol = frag_mol
        data.link_mol = link_mol
        
        self.processed_data = data
        return data
        
class MolReconsError(Exception):
    pass



def bfs(nbh_list, node, k=2, valid_list=[]):
    visited = [node]
    queue = [node]
    level = [0]
    bfs_perm = []

    while len(queue) > 0:
        m = queue.pop(0)
        l = level.pop(0)
        if l > k:
            break
        bfs_perm.append(m)

        for neighbour in nbh_list[m]:
            if neighbour not in visited and neighbour in valid_list:
                visited.append(neighbour)
                queue.append(neighbour)
                level.append(l + 1)
    return bfs_perm



def reconstruct_from_generated_with_edges(data, raise_error=True, sanitize=True):
    xyz = data.pos.clone().cpu().tolist()
    atomic_nums = data.element.clone().cpu().tolist()
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()
    bond_index = data.bond_index.clone().cpu().tolist()
    bond_type = data.bond_type.clone().cpu().tolist()
    n_atoms = len(atomic_nums)

    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)
    
    # add atoms and coordinates
    for i, atom in enumerate(atomic_nums):
        rd_atom = Chem.Atom(atom)
        rd_mol.AddAtom(rd_atom)
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    
    # add bonds
    bond_length_dict = {}
    for i, type_this in enumerate(bond_type):
        node_i, node_j = bond_index[0][i], bond_index[1][i]
        if node_i < node_j:
            if type_this == 1:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 4:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
            else:
                raise Exception('unknown bond order {}'.format(type_this))
            bond_length_dict[(node_i, node_j)] = rd_conf.GetAtomPosition(node_i).Distance(rd_conf.GetAtomPosition(node_j))
            
    # sanitize bond        
        for bond in rd_mol.GetBonds():
            node_i = bond.GetBeginAtomIdx()
            node_j = bond.GetEndAtomIdx()
            bond_length = bond_length_dict[(node_i, node_j)]
            bond_center = (rd_conf.GetAtomPosition(node_i) + rd_conf.GetAtomPosition(node_j)) / 2
            bond_vector = rd_conf.GetAtomPosition(node_j) - rd_conf.GetAtomPosition(node_i)
            bond_vector.Normalize()
            bond_length /= 2
            bond_center += bond_vector * bond_length
            bond.SetBondDir(Chem.BondDir.NONE)
            bond.GetStereo()
    
    # modify
    try:
        rd_mol = modify_submol(rd_mol)
    except:
        if raise_error:
            raise MolReconsError()
        else:
            print('MolReconsError')
            
    # check valid
    try:
        rd_mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol))
        if rd_mol_check is None:
            if raise_error:
                raise MolReconsError()
            else:
                print('MolReconsError')
    except: pass
    
    rd_mol = rd_mol.GetMol()
    # Kekulize
    try:
        if 4 in bond_type:  # mol may directlu come from ture mols and contains aromatic bonds
            Chem.Kekulize(rd_mol, clearAromaticFlags=True)
        if sanitize:
            try:
                Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_SETAROMATICITY)
            except:
                Chem.SanitizeMol(rd_mol)
    except: pass
    
    try:
        Chem.MolToMolFile(rd_mol, '/datapool/data2/home/pengxingang/zhangyijia/testing_mol.sdf')
    except: pass
    return rd_mol

def modify_submol(mol):  # modify mols containing C=N(C)O
    submol = Chem.MolFromSmiles('C=N(C)O', sanitize=False)
    sub_fragments = mol.GetSubstructMatches(submol)
    for fragment in sub_fragments:
        atomic_nums = np.array([mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in fragment])
        idx_atom_N = fragment[np.where(atomic_nums == 7)[0][0]]
        idx_atom_O = fragment[np.where(atomic_nums == 8)[0][0]]
        mol.GetAtomWithIdx(idx_atom_N).SetFormalCharge(1)  # set N to N+
        mol.GetAtomWithIdx(idx_atom_O).SetFormalCharge(-1)  # set O to O-
    return mol

if __name__ == '__main__':
    fragment_1_path = '/datapool/data2/home/pengxingang/zhangyijia/test_fragment/PTPN2/PTPN2_idx_1_7uad/7uad_rec_fixed/_idx_8_fragment_without_add_ring__CC(O)(C(=O)O)C(F)(F)F_ebc.sdf'
    fragment_2_path = '/datapool/data2/home/pengxingang/zhangyijia/test_fragment/PTPN2/PTPN2_idx_1_7uad/7uad_rec_fixed/_idx_9_fragment_without_add_ring__O=C(O)C(O)(O)C(F)(F)F_xyg.sdf'
    data = GenerateLinkerDataset(fragment_1_path, fragment_2_path)
