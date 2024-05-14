import argparse
import os
import sys
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
import random
from copy import deepcopy

sys.path.append('.')
belong_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(belong_dir)

import utils.misc as misc
import utils.transforms as trans

from datasets.linker_dataset import get_linker_dataset
from models.diff_protac_bond import DiffPROTACModel
from utils.reconstruct_linker import parse_sampling_result, parse_sampling_result_with_bond
from torch_geometric.data import Batch
from utils.evaluation import eval_success_rate
from utils.prior_num_atoms import setup_configs, sample_atom_num

if __name__ == '__main__':
    ### basic settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/datapool/data2/home/pengxingang/zhangyijia/LinkerNet/configs/sampling/zinc.yml')
    parser.add_argument('--ckpt_path', type=str,default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='./outputs/zinc'
                        ,help='output_folder_path') 
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--save_traj', type=eval, default=False)
    parser.add_argument('--cand_anchors_mode', type=str, default='k-hop', choices=['k-hop', 'exact'])
    parser.add_argument('--cand_anchors_k', type=int, default=2)
    #parser.add_argument('--geometry_mode', type=str, default='distance_and_two_rot' #None
    #                    ,help='class trans.RelativeGeometry.mode Determine whether to use the original fragment position info or not')
    ### logger files have been deleted
    
    ### add fragment_paths
    parser.add_argument('--fragment_1_path', type=str,  default='/datapool/data2/home/pengxingang/zhangyijia/test_fragment_22.sdf'
                        ,help='the path of given fragment id=1')
    parser.add_argument('--fragment_2_path', type=str,  default='/datapool/data2/home/pengxingang/zhangyijia/test_fragment_277.sdf'
                        ,help='the path of given fragment id=2')
    args = parser.parse_args()
    
    ### disable Rdkit warnings
    RDLogger.DisableLog('rdApp.*')
    
    ### load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.sample.seed)
    if args.fragment_1_path is not None and args.fragment_2_path is not None:
        fragment_1_name = os.path.basename(args.fragment_1_path)
        fragment_2_name = os.path.basename(args.fragment_2_path)
        tag_1, tag_2 = fragment_1_name.split('.')[0], fragment_2_name.split('.')[0]
        tag = str(tag_1 + '_' + tag_2)      # TEST here determines the folder name. Which can be changed for different tasks
    elif args.fragment_1_path is not None and args.fragment_2_path is None:
        tag = os.path.basename(args.fragment_1_path).split('.')[0]
        
    logger, log_dir = misc.setup_logdir_my(
        config=args.config, logdir=args.outdir, tag=args.tag if args.tag is not None else tag)
    #logger.info(args)
    
    ### select running-on device
    args.device = args.device if torch.cuda.is_available() else torch.device('cpu')
    
    ### select geometry mode
    #args.geometry_mode = args.geometry_mode if args.geometry_mode is not None else 'two_pos_and_rot'
    # Load checkpoint
    ckpt_path = config.model.checkpoint if args.ckpt_path is None else args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=args.device)
    #logger.info(f'Successfully load the model! {ckpt_path}')
    cfg_model = ckpt['configs'].model

    # Transforms INFO        ## FeaturizeAtom class does not need to be changed
    test_transform = Compose([
        trans.FeaturizeAtom(config.dataset.name, add_atom_feat=False, add_atom_type=True)
    ])
    atom_featurizer = trans.FeaturizeAtom(
        config.dataset.name, known_anchor=cfg_model.known_anchor, add_atom_type=False, add_atom_feat=True)
    
    graph_builder = trans.BuildCompleteGraph(known_linker_bond=cfg_model.known_linker_bond,
                                             known_cand_anchors=config.sample.get('cand_bond_mask', False))
    init_transform_1 = Compose([
        atom_featurizer,
        trans.SelectCandAnchors(mode=args.cand_anchors_mode, k=args.cand_anchors_k),
        graph_builder,
        trans.StackFragLocalPos(max_num_atoms=config.dataset.get('max_num_atoms', 30)),
        trans.RelativeGeometry(mode=cfg_model.get('rel_geometry', 'distance_and_two_rot'))
    ])
    init_transform_2 = Compose([
        atom_featurizer,
        trans.SelectCandAnchors(mode='exact', k=args.cand_anchors_k),
        graph_builder,
        trans.StackFragLocalPos(max_num_atoms=config.dataset.get('max_num_atoms', 30)),
        trans.RelativeGeometry(mode=cfg_model.get('rel_geometry', 'distance_and_two_rot'))
    ])
    

    FOLLOW_BATCH = ['edge_type']
    COLLATE_EXCLUDE_KEYS = ['nbh_list']

    # Model
    #logger.info('Building model...')
    model = DiffPROTACModel(
        cfg_model,
        num_classes=atom_featurizer.num_classes,
        num_bond_classes=graph_builder.num_bond_classes,
        atom_feature_dim=atom_featurizer.feature_dim,
        edge_feature_dim=graph_builder.bond_feature_dim
    ).to(args.device)
    #logger.info('Num of parameters is %.2f M' % (np.sum([p.numel() for p in model.parameters()]) / 1e6))
    model.load_state_dict(ckpt['model'])
    #logger.info(f'Load model weights done!')
    
    ### Exists data loading
    if os.path.exists(args.fragment_1_path) and args.fragment_2_path is not None:  ## given two fragments
        if os.path.exists(args.fragment_2_path) and str(args.fragment_1_path).endswith('.sdf') and str(args.fragment_2_path).endswith('.sdf'):
            CreateDataset = trans.GenerateLinkerDataset(args.fragment_1_path, args.fragment_2_path)
            raw_data = CreateDataset._get_data
            atom_indices_f1 = CreateDataset._get_indices_1
            atom_indices_f2 = CreateDataset._get_indices_2
            #logger.info(f'Generate Basic Data Successfully')
            given_parts = 2
        else:
            print(f'No Such Fragment SDF File. Please Check if There is any spelling mistake')
    elif args.fragment_2_path is None and os.path.exists(args.fragment_1_path):         ## given one original molecule
        if str(args.fragment_1_path).endswith('.sdf'):
            CreateDataset = trans.GenerateLinkerDataset(args.fragment_1_path)
            raw_data = CreateDataset._get_data
            atom_indices_f1 = CreateDataset._get_indices_1
            atom_indices_f2 = CreateDataset._get_indices_2
            #logger.info(f'Generate Basic Data Successfully')
            given_parts = 1
        else:
            print(f'No Such Molecule SDF File. Please Check if There is any spelling mistake')
            
    # Sampling
    assert config.sample.num_atoms in ['ref', 'prior'] ### 'prior' means exact anchor known Zinc dataset file
    if config.sample.num_atoms == 'prior':
        num_atoms_config = setup_configs(mode='frag_center_distance')
    else:
        num_atoms_config = None    
        
    # get data list
    data_list = [deepcopy(raw_data) for _ in range(args.num_samples)]
    
    # modify data list
    if num_atoms_config is None: ## 'ref' mode. means known nothing about the anchor/linker_length
        new_data_list = []
        number_min, number_max = 0, 6
        select_linker_atom_num_list = np.arange(number_max).tolist()
        for data in data_list:
            if given_parts == 2:
                if args.num_samples >= 6:
                    sample_linker_atoms = random.randint(number_min, number_max)
                else:
                    random.shuffle(select_linker_atom_num_list)
                    sample_linker_atoms = select_linker_atom_num_list[0]
                    del select_linker_atom_num_list[0]

                print(f'Selected Linker Atoms: {sample_linker_atoms}')
                
            elif given_parts == 1:
                sample_linker_atoms = raw_data['linker_nums']
        
            ### process GenerateLinkerDataset
            ## add anchor info
            RandomAnchor = trans.GetRandomAnchor(data)
            RandomAnchor._get_anchor()
            data = RandomAnchor.get_data
            ## random rotation process
            RandomDataset = trans.RandomFragLinkerDataset(data)
            ChangeToFragLinkerData = RandomDataset._rotation()
            ChangeToFragLinkerData = test_transform(ChangeToFragLinkerData)
            print(f'ChangeToFragLinkerData.anchor_indices: {ChangeToFragLinkerData.anchor_indices}')  # TEST
            num_f1_atoms = len(ChangeToFragLinkerData.atom_indices_f1)
            num_f2_atoms = len(ChangeToFragLinkerData.atom_indices_f2)
            num_f_atoms = num_f1_atoms + num_f2_atoms
                
            ### fragment information
            frag_pos = ChangeToFragLinkerData.pos[ChangeToFragLinkerData.fragment_mask > 0]
            frag_atom_type = ChangeToFragLinkerData.atom_type[ChangeToFragLinkerData.fragment_mask > 0]
            frag_bond_idx = (ChangeToFragLinkerData.bond_index[0] < num_f_atoms) & (ChangeToFragLinkerData.bond_index[1] < num_f_atoms)
            ChangeToFragLinkerData.fragment_mask = torch.LongTensor([1] * num_f1_atoms + [2] * num_f2_atoms + [0] * sample_linker_atoms)
            ChangeToFragLinkerData.linker_mask = (ChangeToFragLinkerData.fragment_mask == 0)
            ChangeToFragLinkerData.anchor_mask = torch.cat([ChangeToFragLinkerData.anchor_mask[:num_f_atoms], torch.zeros([sample_linker_atoms]).long()])
            ChangeToFragLinkerData.bond_index = ChangeToFragLinkerData.bond_index[:, frag_bond_idx]
            ChangeToFragLinkerData.bond_type = ChangeToFragLinkerData.bond_type[frag_bond_idx]
            ChangeToFragLinkerData.pos = torch.cat([frag_pos, torch.zeros([sample_linker_atoms, 3])], dim=0)
            ChangeToFragLinkerData.atom_type = torch.cat([frag_atom_type, torch.zeros([sample_linker_atoms]).long()])
            new_data = init_transform_1(ChangeToFragLinkerData) if given_parts == 2 else init_transform_2(ChangeToFragLinkerData)
            new_data_list.append(new_data)
                                       
    batch = Batch.from_data_list(
        new_data_list, follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS).to(args.device)
    traj_batch, final_x, final_c, final_bond = model.sample(
        batch,
        p_init_mode=cfg_model.frag_pos_prior,
        guidance_opt=config.sample.guidance_opt
    )
    if model.train_bond:
        gen_mols = parse_sampling_result_with_bond(
            new_data_list, final_x, final_c, final_bond, atom_featurizer,
            known_linker_bonds=cfg_model.known_linker_bond, check_validity=True)
    else:
        gen_mols = parse_sampling_result(new_data_list, final_x, final_c, atom_featurizer)

    try:  
        save_path_dirname = log_dir
        for i in range(len(new_data_list)):
            save_path_basename = '%d.sdf' %i
            save_path = os.path.join(save_path_dirname, save_path_basename)
            Chem.MolToMolFile(gen_mols[i], save_path)
    except:
        print(f'Error When Saving the rdmol object {gen_mols[i]} to selected folder')

    logger.info('Sample done!')
    