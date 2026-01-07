# -*- coding: utf-8 -*-
import argparse
import os
import pickle as pkl
import random

import numpy as np
import torch
import torch.distributed
import wandb
from ase.units import GPa

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.forcefield.m3gnet.scaling import AtomScaling
from mattersim.forcefield.potential import Potential
from mattersim.utils.atoms_utils import AtomsAdaptor
from mattersim.utils.logger_utils import get_logger
import datetime

# TODO (jiahang): distributed training set gpu device for each process
logger = get_logger()
local_rank = int(os.environ.get("LOCAL_RANK", 0))

def main(args):
    if args.distributed:
        if args.device == "cuda":
            torch.distributed.init_process_group(backend="nccl")
        else:
            torch.distributed.init_process_group(backend="gloo")
    args_dict = vars(args)
    if args.wandb and local_rank == 0:
        # wandb_api_key = (
        #     args.wandb_api_key
        #     if args.wandb_api_key is not None
        #     else os.getenv("WANDB_API_KEY")
        # )
        # wandb.login(key=wandb_api_key)
        # use current timestamp as suffix
        run_name = args.wandb_project + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_dir = os.path.join(args.wandb_dir, run_name)
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=args,
            mode="offline",
            dir=wandb_dir
            # id=args.run_name,
            # resume="allow",
        )
        # parent dir of run.dir
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(run.dir))) # jiahang: dirty!
        # create checkpoint save path under parent dir
        args.save_path = os.path.join(parent_dir, 'checkpoints')
        os.makedirs(args.save_path, exist_ok=True)

    if args.wandb:
        args_dict["wandb"] = wandb

    if args.distributed:
        torch.distributed.barrier()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train_data_path.endswith(".pkl"):
        with open(args.train_data_path, "rb") as f:
            atoms_train = pkl.load(f)
    else:
        atoms_train = AtomsAdaptor.from_file(filename=args.train_data_path)
    energies = []
    forces = [] if args.include_forces else None
    stresses = [] if args.include_stresses else None
    logger.info("Processing training datasets...")
    for atoms in atoms_train:
        energies.append(atoms.get_potential_energy())
        if args.include_forces:
            forces.append(atoms.get_forces())
        if args.include_stresses:
            stresses.append(atoms.get_stress(voigt=False) / GPa)  # convert to GPa

    logger.info("Building training dataloader...")
    dataloader = build_dataloader(
        atoms_train,
        energies,
        forces,
        stresses,
        shuffle=True,
        pin_memory=(args.device == "cuda"),
        is_distributed=args.distributed,
        multiprocessing=4,
        **args_dict,
    )

    device = args.device
    scale = None
    if args.normalize:
        logger.info("Calculating normalization factors from training data...")
        scale = AtomScaling(
            atoms=atoms_train,
            total_energy=energies,
            forces=forces,
            verbose=True,
            **args_dict,
        ).to(device)

    if args.valid_data_path is not None:
        if args.valid_data_path.endswith(".pkl"):
            with open(args.valid_data_path, "rb") as f:
                atoms_val = pkl.load(f)
        else:
            atoms_val = AtomsAdaptor.from_file(filename=args.valid_data_path)
        energies_val = []
        forces_val = [] if args.include_forces else None
        stresses_val = [] if args.include_stresses else None
        logger.info("Processing validation datasets...")
        for atoms in atoms_val:
            energies_val.append(atoms.get_potential_energy())
            if args.include_forces:
                forces_val.append(atoms.get_forces())
            if args.include_stresses:
                stresses_val.append(
                    atoms.get_stress(voigt=False) / GPa
                )  # convert to GPa
        val_dataloader = build_dataloader(
            atoms_val,
            energies_val,
            forces_val,
            stresses_val,
            pin_memory=(args.device == "cuda"),
            is_distributed=args.distributed,
            **args_dict,
        )
    else:
        val_dataloader = None

    # --- Model Initialization: From Scratch vs. Fine-tuning ---
    if args.load_model_path:
        logger.info(
            f"Mode: Fine-tuning. Loading pre-trained model from {args.load_model_path}."
        )
        potential = Potential.from_checkpoint(
            load_path=args.load_model_path,
            load_training_state=False,
            **args_dict,
        )
    else:
        logger.info("Mode: Training from scratch. Initializing a new M3Gnet model.")
        model = M3Gnet(
            max_n=args.max_n,
            max_l=args.max_l,
            n_blocks=args.n_blocks,
            units=args.units,
            cutoff=args.cutoff,
            threebody_cutoff=args.threebody_cutoff,
            n_atom_types=args.n_atom_types,
        ).to(device)
        potential = Potential(model=model, **args_dict)

    if args.normalize and scale is not None:
        logger.info("Applying data normalizer to the model.")
        potential.model.set_normalizer(scale)

    if args.distributed:
        if args.device == "cuda":
            potential.model = torch.nn.parallel.DistributedDataParallel(potential.model)
        torch.distributed.barrier()

    potential.train_model(
        dataloader,
        val_dataloader,
        loss=torch.nn.HuberLoss(delta=0.01),
        is_distributed=args.distributed,
        **args_dict,
    )

    # if local_rank == 0 and args.save_checkpoint and args.wandb:
    #     wandb.save(os.path.join(args.save_path, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path parameters
    parser.add_argument(
        "--distributed", action="store_true", help="Whether to use distributed training"
    )
    parser.add_argument(
        "--train_data_path", type=str, default="./sample.xyz", help="train data path"
    )
    parser.add_argument(
        "--valid_data_path", type=str, default=None, help="valid data path"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Path to a pre-trained model for fine-tuning. If not provided, trains a new model from scratch.",  # noqa: E501
    )
    parser.add_argument(
        "--save_checkpoint",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=10,
        help="save checkpoint every ckpt_interval epochs",
    )
    parser.add_argument("--device", type=str, default="cuda")

    # model parameters
    parser.add_argument(
        "--cutoff", type=float, default=5.0, help="Two-body cutoff radius"
    )
    parser.add_argument(
        "--threebody_cutoff",
        type=float,
        default=4.0,
        help="Three-body cutoff radius, smaller than two-body cutoff",
    )
    # New arguments for training from scratch
    parser.add_argument(
        "--n_atom_types",
        type=int,
        default=119,
        help="Number of atom types (atomic numbers from 1 to n_atom_types)",
    )
    parser.add_argument(
        "--max_n", type=int, default=3, help="Max value of n for radial basis"
    )
    parser.add_argument(
        "--max_l", type=int, default=3, help="Max value of l for spherical harmonics"
    )
    parser.add_argument(
        "--n_blocks", type=int, default=3, help="Number of interaction blocks"
    )
    parser.add_argument(
        "--units", type=int, default=64, help="Number of units in hidden layers"
    )

    # training parameters
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="step epoch for learning rate scheduler",
    )
    parser.add_argument(
        "--include_forces",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--include_stresses",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--force_loss_ratio", type=float, default=1.0)
    parser.add_argument("--stress_loss_ratio", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # scaling parameters
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Normalize energy and forces based on training data. Recommended for both training from scratch and fine-tuning.",  # noqa: E501
    )
    parser.add_argument("--scale_key", type=str, default="per_species_forces_rms")
    parser.add_argument(
        "--shift_key", type=str, default="per_species_energy_mean_linear_reg"
    )
    parser.add_argument("--init_scale", type=float, default=None)
    parser.add_argument("--init_shift", type=float, default=None)
    parser.add_argument(
        "--trainable_scale",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--trainable_shift",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    # wandb parameters
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="wandb_test")
    parser.add_argument("--wandb_dir", type=str, default="./wandb_logs")
    args = parser.parse_args()
    main(args)
