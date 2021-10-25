import sys
import argparse
import pprint
import torch
from torch.nn.functional import threshold
from utils import str2bool
import wandb
from sweep import get_sweep_id
# sweep_id = get_sweep_id('grid')
# wandb.agent(sweep_id=sweep_id)


# config = wandb.config
# config.learning_rate = 0.01

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument('--ckpt_path',
                       required=False,
                       help="./ckpoint1",
                       type=str,
                       default="./ckpoint1")
# argparser.add_argument('--ckpt_path',
#                        required=False,
#                        help="./ckpoint",
#                        type=str,
#                        default="/Users/seojiwon/sol_project_twigfarm/ckpoint")
# argparser.add_argument('--clf_ckpt_path',
#                        help="path to load pretrained classifier",
#                        type=str,
#                        default="/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/bert_pretrained")
# argparser.add_argument('--clf_ckpt_path',
#                        help="path to load pretrained classifier",
#                        type=str,
#                        default="/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/AIhub_clf.pt")
# # dataloading
argparser.add_argument('--clf_ckpt_path',
                       help="path to load pretrained classifier, check your clf model",
                       type=str,
                    #    default="/home/lab12/sol/sol_project_twigfarm/data_bert/br.pt") #bert clf model
                       default="/home/lab12/sol/sol_project_twigfarm/NSMC-last.ckpt") #bart clf model 
#
#                        default="/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/AIhub_clf.pt")


argparser.add_argument("--clf_model",
                        help = "select clf model between bert and bart",
                        type =  str, 
                        choices = [ "bert", "bart"],
                        default = "bart")
argparser.add_argument('--dataset',
                       type=str,
                       choices=['yelp', 'nsmc', 'AIhub','br'],
                       default="br")
argparser.add_argument('--text_file_path',
                       type=str)
argparser.add_argument('--val_text_file_path',
                       type=str)
argparser.add_argument('--batch_size',
                       type=int,
                       default=64)

# argparser.add_argument('--batch_size',
#                        type=int,
#                        default=wandb.config.batch_size if "batch_size" in wandb.config.keys() and isinstance(wandb.config.batch_size, int) else 64)
argparser.add_argument('--max_seq_length',
                       type=int,
                       default=64)
# argparser.add_argument('--max_seq_length',
#                        type=int,
#                          default=wandb.config.max_seq_length if "max_seq_length" in wandb.config.keys() and isinstance(wandb.config.max_seq_length, int) else 64)

argparser.add_argument('--num_workers',
                       type=int,
                       default=4)

# architecture
# argparser.add_argument('--dim_y',
#                        type=int,
#                        default=wandb.config.dim_y if "dim_y" in wandb.config.keys() and isinstance(wandb.config.dim_y, int) else 200)

# argparser.add_argument('--dim_z',
#                        type=int,
#                        default=wandb.config.dim_z if "dim_z" in wandb.config.keys() and isinstance(wandb.config.dim_z, int) else 500)
argparser.add_argument('--dim_y',
                       type=int,
                       default= 200)

argparser.add_argument('--dim_z',
                       type=int,
                       default = 500)

argparser.add_argument('--dim_emb',
                       type=int,
                       default=768)
argparser.add_argument('--filter_sizes',
                       type=int,
                       nargs='+',
                       default=[1, 2, 3, 4, 5])
argparser.add_argument('--n_filters',
                       type=int,
                       default=128)

# learning
argparser.add_argument('--epochs',
                       type=int,
                       default=20)
argparser.add_argument('--weight_decay',
                       type=float,
                       default=0.0)
# argparser.add_argument('--weight_decay',
#                        type=float,
#                        default=wandb.config.weight_decay if "weight_decay" in wandb.config.keys() and isinstance(wandb.config.weight_decay, float) else 0.0)

argparser.add_argument('--max_grad_norm',
                       type=float,
                       default=1.0)
# argparser.add_argument('--max_grad_norm',
#                        type=float,
#                        default=wandb.config.max_grad_norm if "max_grad_norm" in wandb.config.keys() and isinstance(wandb.config.max_grad_norm, float) else 1.0)

argparser.add_argument('--lr',
                       type=float,
                       default=5e-4)

# argparser.add_argument('--lr',
#                        type=float,
#                        default=wandb.config.lr if "lr" in wandb.config.keys() and isinstance(wandb.config.lr, float) else 5e-4)

argparser.add_argument('--disc_lr',
                       type=float,
                       default= 5e-5)

# argparser.add_argument('--disc_lr',
#                        type=float,
#                        default=wandb.config.disc_lr if "disc_lr" in wandb.config.keys() and isinstance(wandb.config.disc_lr, float) else 5e-5)

argparser.add_argument("--temperature",
                       type=float,
                       default=0.1)

# argparser.add_argument("--temperature",
#                        type=float,
#                        default=wandb.config.temperature if "temperature" in wandb.config.keys() and isinstance(wandb.config.temperature, float) else 0.1)

argparser.add_argument('--use_gumbel',
                       default=True,
                       type=str2bool)
argparser.add_argument('--rho',  # loss_rec + rho * loss_adv
                       type=float,
                       default=1)
argparser.add_argument('--two_stage',
                        type=str2bool,
                        default=False)
argparser.add_argument('--second_stage_num',
                        type=int,
                        default=2)
argparser.add_argument('--gan_type',
                       default='wgan-gp',
                       choices=['vanilla', 'wgan-gp', 'lsgan'])
argparser.add_argument('--gp_weight',
                       default=1.0,
                       type=float)
argparser.add_argument('--log_interval',
                       default=100,
                       type=int)
argparser.add_argument('--language',
                       default='ko',
                       choices=['ko', 'en'])
argparser.add_argument('--threshold',
                       type=float,
                       default=None)
# argparser.add_argument('--threshold',
#                        type=float,
#                        default=wandb.config.threshold if "threshold" in wandb.config.keys() and isinstance(wandb.config.threshold, float) else None)

# testing
argparser.add_argument('--mode',
                        help='train or transfer',
                        choices=['train', 'transfer', 'interactive'],
                        default='train',
                        type=str)
argparser.add_argument('--test_text_path',
                       help='path to text file whose each line contains one sentence',
                       default=None)
argparser.add_argument('--transfer_to', #only use in interactive transfer
                       default=1,
                       type=int,
                       choices=[0, 1])
argparser.add_argument('--n_samples',
                       default=1000,
                       type=int)
argparser.add_argument('--transfer_max_len',
                       default=64,
                       type=int)
argparser.add_argument("--transfer_result_save_path",
                       default=None,
                       help="path to save transfer result")

# others
argparser.add_argument("--cuda_device",
                       type=int,
                       default=0)
argparser.add_argument("--load_ckpt", # Tobe deleted after experimenting
                       type=str,
                       default="ckpts/wgan_no_threshold.pt")

args = argparser.parse_args()

args.device = torch.device(
    'cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu'
)

# dataset presets
if args.dataset == 'yelp':
    args.language = 'en'
    args.text_file_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/yelp/yelp.sentiment.train'
    args.val_text_file_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/yelp/yelp.sentiment.val'
    args.test_text_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/yelp/yelp.sentiment.test'
    args.clf_ckpt_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/yelp_clf.pt'
elif args.dataset == 'nsmc':
    args.language = 'ko'
    args.text_file_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/nsmc/ratings_train.txt'
    args.val_text_file_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/nsmc/ratings_test.txt'
    args.test_text_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/nsmc/ratings_test.txt'
    args.clf_ckpt_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/nsmc_clf.pt'
elif args.dataset == 'AIhub':
    args.language = 'ko'
    args.text_file_path = "/home/tf-dev-02/workspace_sol/style-transfer/NLP_text-style-transfer/data_bert/AIhub_train.csv"
    args.val_text_file_path = "/home/tf-dev-02/workspace_sol/style-transfer/NLP_text-style-transfer/data_bert/AIhub_test.csv"
    args.test_text_path = "/home/tf-dev-02/workspace_sol/style-transfer/NLP_text-style-transfer/data_bert/AIhub_train.csv"
    # args.clf_ckpt_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/aihub_clf.pt'
    args.clf_ckpt_path =  "/home/tf-dev-02/workspace_sol/outputs/aihub2.pt"
    # args.clf_ckpt_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/nsmc_clf.pt'
elif args.dataset == 'br':
    args.language = 'ko'
    args.text_file_path = "/home/lab12/sol/sol_project_twigfarm/data_bert/train_data.txt"
    args.val_text_file_path = '/home/lab12/sol/sol_project_twigfarm/data_bert/test_data.txt'
    args.test_text_path = "/home/lab12/sol/sol_project_twigfarm/data_bert/val_data.txt"
    # args.clf_ckpt_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/aihub_clf.pt'
    args.clf_ckpt_path = "/home/lab12/sol/sol_project_twigfarm/NSMC-last.ckpt"



elif args.dataset is None:
    assert args.text_file_path is not None
    assert args.val_text_file_path is not None
    assert args.test_text_path is not None

print('------------------------------------------------')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))
print('------------------------------------------------')
