import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--exp_name", help="Experiment name", type=str, required=True)
parser.add_argument("--verbose", help="Increase output verbosity", action="store_true")
parser.add_argument("--data_dir", help="Base directory of the dataset. Check README for the dataset structure.", type=str, required=True)
parser.add_argument("--colmap", help="Uses COLMAP camera parameters instead of the ground truth. If it can't find parameters it runs COLMAP first. If not given use params.pk file", 
                    action="store_true")
parser.add_argument("--near", help="Near value of the cameras. Utilized only if --colmap is not given.", type=float)
parser.add_argument("--far", help="Far value of the cameras. Utilized only if --colmap is not given", type=float)
parser.add_argument("--img_prefix", help="Prefix of image names", default='frame.', type=str)
parser.add_argument("--img_format", help="Format of the images", default=".png", type=str)
parser.add_argument("--N_coarse", help="Number of coarse samples", default=32, type=int)
parser.add_argument("--N_fine", help="Number of fine samples. Set to 0 to disable hierarchical sampling", default=64, type=int)
parser.add_argument("--Lp", help="Number of position embedding bases", default=10, type=int)
parser.add_argument("--Ld", help="Number of direciton embedding bases", default=4, type=int)
parser.add_argument("--batch_size", help="Number of rays for each training step", default=4096, type=int)
parser.add_argument("--n_layer", help="Number of layers", default=10, type=int)
parser.add_argument("--skip_layer", help="Which layer the input should skip", default=5, type=int)
parser.add_argument("--n_hidden", help="Number of hidden neurons at each layer", default=256, type=int)
parser.add_argument("--n_epochs", help="Number of epochs", default=30, type=int)
parser.add_argument("--down_scale", help="Scale down the images", default=1, type=int)
parser.add_argument("--initial_lr", help="Initial learning rate", default=5e-4, type=float)
parser.add_argument("--lr_decay", help="Decay lr by 0.1 every n/1000 steps", default=250, type=int)
parser.add_argument("--eval_step", help="Evaluation step", default=1e3, type=int)
parser.add_argument("--no_log", help="Disable logging", action="store_true")