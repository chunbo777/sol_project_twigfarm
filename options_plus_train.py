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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import time
import itertools
import sys, os

from transformers import BertModel, ElectraModel, ElectraTokenizer
sys.path.insert(0, "/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/")
from dataloader import get_dataloader_for_style_transfer
from model import Encoder, Generator, Discriminator
from bert_pretrained import bert_tokenizer, get_bert_word_embedding, FILE_ID
from bert_pretrained.classifier import BertClassifier
from loss import loss_fn, gradient_penalty
from evaluate import calculate_accuracy, calculate_frechet_distance
from transfer import style_transfer


from utils import AverageMeter, ProgressMeter, download_google, Metric_Printer
""" 
wandb(weight and biases) 사용법
wandb에 계정 생성 및 로그인 => 하이퍼 파라미터 튜닝을 위한 모듈 sweep를 사용할 것이기 때문에 
sweepid를 통한 메인호출을 하기위해 get_sweep_id함수를 임포트(sweep.py) sweep_id를 통해 agent를 디파인
옵션에 있는 여러 파라미터들을 wandb의 config를 인용한 파라미터로 변경   

"""

import wandb
wandb.login()
#wandb를 선행 호출 후 options를 임포트해야 오류 발생안함

# from sweep import get_sweep_id
from sweep import get_sweep_id

from options import args
class Trainer:
    def __init__(self):
        # get models
        embedding = get_bert_word_embedding()
        if os.path.isfile(args.load_ckpt):
            self.models = torch.load(args.load_ckpt)
        else:
            self.models = nn.ModuleDict({
                'embedding': embedding,
                'encoder': Encoder(embedding, args.dim_y, args.dim_z),
                'generator': Generator(
                    embedding, args.dim_y, args.dim_z, args.temperature,
                    bert_tokenizer.bos_token_id, use_gumbel=args.use_gumbel
                ),
                'disc_0': Discriminator(  # 0: real, 1: fake
                    args.dim_y + args.dim_z, args.n_filters, args.filter_sizes
                ),
                'disc_1': Discriminator(  # 1: real, 0: fake
                    args.dim_y + args.dim_z, args.n_filters, args.filter_sizes
                ),
            })
        self.models.to(args.device)
        wandb.watch(models)
        # pretrained classifier
        self.clf = BertClassifier()
        if args.clf_ckpt_path is not None:
            download_google(FILE_ID, args.clf_ckpt_path)
            ckpt = torch.load(
                args.clf_ckpt_path,
                map_location=lambda storage, loc: storage
            )
            self.clf.load_state_dict(ckpt['model_state_dict'])
        self.clf.to(args.device)
        self.clf.eval()

        # get dataloaders
        self.train_loaders = get_dataloader_for_style_transfer(
            args.text_file_path, shuffle=True, drop_last=True
        )
        # label placeholders
        self.zeros = torch.zeros(args.batch_size, 1).to(args.device)
        self.ones = torch.ones(args.batch_size, 1).to(args.device)

        # get optimizers
        self.optimizer = optim.AdamW(
            list(itertools.chain.from_iterable([
                list(self.models[k].parameters())
                for k in ['embedding', 'encoder', 'generator']
            ])),
            lr=args.lr,
            betas=(0.5, 0.9),
            weight_decay=args.weight_decay
        )
        self.disc_optimizer = optim.AdamW(
            list(itertools.chain.from_iterable([
                list(self.models[k].parameters())
                for k in ['disc_0', 'disc_1']
            ])),
            lr=args.disc_lr,
            betas=(0.5, 0.9),
            weight_decay=args.weight_decay
        )

        self.epoch = 0

    def train_epoch(self):
        self.models.train()
        self.epoch += 1

        # record training statistics
        avg_meters = {
            'loss_rec': AverageMeter('Loss Rec', ':.4e'),
            'loss_adv': AverageMeter('Loss Adv', ':.4e'),
            'loss_disc': AverageMeter('Loss Disc', ':.4e'),
            'time': AverageMeter('Time', ':6.3f')
        }
        for key, value in avg_meters.items():
            wandb.log({key : value})
        progress_meter = ProgressMeter(
            len(self.train_loaders[0]),
            avg_meters.values(),
            prefix="Epoch: [{}]".format(self.epoch)
        )

        # begin training from minibatches
        for ix, (data_0, data_1) in enumerate(zip(*self.train_loaders)):
            start_time = time.time()

            # load text and labels
            src_0, src_len_0, labels_0 = data_0
            src_0, labels_0 = src_0.to(args.device), labels_0.to(args.device)
            src_1, src_len_1, labels_1 = data_1
            src_1, labels_1 = src_1.to(args.device), labels_1.to(args.device)

            # encode
            encoder = self.models['encoder']
            z_0 = encoder(labels_0, src_0, src_len_0)  # (batch_size, dim_z)
            z_1 = encoder(labels_1, src_1, src_len_1)

            # recon & transfer
            generator = self.models['generator']
            inputs_0 = (z_0, labels_0, src_0)
            h_ori_seq_0, pred_ori_0 = generator(*inputs_0, src_len_0, False)
            h_trans_seq_0_to_1, _ = generator(*inputs_0, src_len_1, True)

            inputs_1 = (z_1, labels_1, src_1)
            h_ori_seq_1, pred_ori_1 = generator(*inputs_1, src_len_1, False)
            h_trans_seq_1_to_0, _ = generator(*inputs_1, src_len_0, True)

            # discriminate real and transfer
            disc_0, disc_1 = self.models['disc_0'], self.models['disc_1']
            d_0_real = disc_0(h_ori_seq_0.detach())  # detached
            d_0_fake = disc_0(h_trans_seq_1_to_0.detach())
            d_1_real = disc_1(h_ori_seq_1.detach())
            d_1_fake = disc_1(h_trans_seq_0_to_1.detach())

            # discriminator loss
            loss_disc = (
                loss_fn(args.gan_type)(d_0_real, self.ones)
                + loss_fn(args.gan_type)(d_0_fake, self.zeros)
                + loss_fn(args.gan_type)(d_1_real, self.ones)
                + loss_fn(args.gan_type)(d_1_fake, self.zeros)
            )
            # gradient penalty
            if args.gan_type == 'wgan-gp':
                loss_disc += args.gp_weight * gradient_penalty(
                    h_ori_seq_0,            # real data for 0
                    h_trans_seq_1_to_0,     # fake data for 0
                    disc_0
                )
                loss_disc += args.gp_weight * gradient_penalty(
                    h_ori_seq_1,            # real data for 1
                    h_trans_seq_0_to_1,     # fake data for 1
                    disc_1
                )
            avg_meters['loss_disc'].update(loss_disc.item(), src_0.size(0))

            self.disc_optimizer.zero_grad()
            loss_disc.backward()
            self.disc_optimizer.step()

            # reconstruction loss
            loss_rec = (
                F.cross_entropy(    # Recon 0 -> 0
                    pred_ori_0.view(-1, pred_ori_0.size(-1)),
                    src_0[1:].view(-1),
                    ignore_index=bert_tokenizer.pad_token_id,
                    reduction='sum'
                )
                + F.cross_entropy(  # Recon 1 -> 1
                    pred_ori_1.view(-1, pred_ori_1.size(-1)),
                    src_1[1:].view(-1),
                    ignore_index=bert_tokenizer.pad_token_id,
                    reduction='sum'
                )
            ) / (2.0 * args.batch_size)  # match scale with the orginal paper
            avg_meters['loss_rec'].update(loss_rec.item(), src_0.size(0))

            # generator loss
            d_0_fake = disc_0(h_trans_seq_1_to_0)  # not detached
            d_1_fake = disc_1(h_trans_seq_0_to_1)
            loss_adv = (
                loss_fn(args.gan_type, disc=False)(d_0_fake, self.ones)
                + loss_fn(args.gan_type, disc=False)(d_1_fake, self.ones)
            ) / 2.0  # match scale with the original paper
            avg_meters['loss_adv'].update(loss_adv.item(), src_0.size(0))

            # XXX: threshold for training stability
            if (not args.two_stage):
                if (args.threshold is not None
                        and loss_disc < args.threshold):
                    loss = loss_rec + args.rho * loss_adv
                else:
                    loss = loss_rec
            else: # two_stage training
                if (args.second_stage_num > args.epochs-self.epoch): 
                    # last second_stage; flow loss_adv gradients
                    loss = loss_rec + args.rho * loss_adv
                else:
                    loss = loss_rec
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meters['time'].update(time.time() - start_time)

            # log progress
            if (ix + 1) % args.log_interval == 0:
                progress_meter.display(ix + 1)

        progress_meter.display(len(self.train_loaders[0]))

    def evaluate(self):
        self.models.eval()
        # generate samples
        inputs0, inputs1, outputs0, outputs1 = style_transfer(
            encoder=self.models['encoder'],
            generator=self.models['generator'],
            text_path=args.val_text_file_path,
            n_samples=args.n_samples
        )

        # display 10 samples for each
        print('=' * 30 + '\nnegative -> positive\n' + '=' * 30 + '\n')
        for original, transfer in zip(inputs0[:10], outputs0[:10]):
            print(original + ' -> ' + transfer + '\n')
        print('=' * 30 + '\npositive -> negative\n' + '=' * 30 + '\n')
        for original, transfer in zip(inputs1[:10], outputs1[:10]):
            print(original + ' -> ' + transfer + '\n')

        print("Evaluation from {} samples".format(args.n_samples))
        fed = (calculate_frechet_distance(inputs1, outputs0)
               + calculate_frechet_distance(inputs0, outputs1))
        print('FED: {:.4f}'.format(fed))

        loss, acc = calculate_accuracy(
            self.clf,
            outputs0 + outputs1,
            torch.cat([
                torch.ones(len(outputs0)),
                torch.zeros(len(outputs1))
            ]).long().to(args.device)
        )
        print('Loss: {:.4f}'.format(loss.item()))
        print('Accuracy: {:.4f}\n'.format(acc.item()))
        return fed, loss.item(), acc.item()


class Translator:
    def __init__(self):
        self.models = torch.load(args.ckpt_path)
    def transfer(self):
        self.models.eval()
        if args.mode == 'interactive':
            args.test_text_path = None
        _, _, _, _ = style_transfer(
            encoder=self.models['encoder'],
            generator=self.models['generator'],
            text_path=args.test_text_path,
            n_samples=args.n_samples
        )


if __name__ == '__main__':
    # wandb.agent(sweep_id=sweep_id, function=main)
    
    if args.mode == 'train':
        wandb.init(project = "nlp_style_transfer")
        sweep_id = get_sweep_id('grid')
        wandb.agent(sweep_id=sweep_id, function=Trainer())
        # trainer = Trainer()
        printer = Metric_Printer('FED', 'Loss', 'Acc')
        
        loss_save = sys.maxsize
        for _ in range(args.epochs):

            trainer.train_epoch()
            fed, loss, acc = trainer.evaluate()
            if loss < loss_save:
                loss_save = loss
                print ("saving model : " + args.ckpt_path)
                torch.save(trainer.models, args.ckpt_path)

            printer.update(fed, loss, acc)
        print(printer)
    else:
        translator = Translator()
        translator.transfer()



config = wandb.config
config.learning_rate = 0.01


argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument('--ckpt_path',
                       required=False,
                       help="./ckpoint",
                       type=str,
                       default="/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/ckpoint/aihub")
# argparser.add_argument('--clf_ckpt_path',
#                        help="path to load pretrained classifier",
#                        type=str,
#                        default="/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/bert_pretrained")
argparser.add_argument('--clf_ckpt_path',
                       help="path to load pretrained classifier",
                       type=str,
                       default="/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/AIhub_clf.pt")
# dataloading
argparser.add_argument('--dataset',
                       type=str,
                       choices=['yelp', 'nsmc', 'AIhub'],
                       default="AIhub")
argparser.add_argument('--text_file_path',
                       type=str)
argparser.add_argument('--val_text_file_path',
                       type=str)
# argparser.add_argument('--batch_size',
#                        type=int,
#                        default=64)

argparser.add_argument('--batch_size',
                       type=int,
                       default=wandb.config.batch_size if "batch_size" in wandb.config.keys() and isinstance(wandb.config.batch_size, int) else 64)
# argparser.add_argument('--max_seq_length',
#                        type=int,
#                        default=64)
argparser.add_argument('--max_seq_length',
                       type=int,
                         default=wandb.config.max_seq_length if "max_seq_length" in wandb.config.keys() and isinstance(wandb.config.max_seq_length, int) else 64)

argparser.add_argument('--num_workers',
                       type=int,
                       default=4)

# architecture
argparser.add_argument('--dim_y',
                       type=int,
                       default=wandb.config.dim_y if "dim_y" in wandb.config.keys() and isinstance(wandb.config.dim_y, int) else 200)

argparser.add_argument('--dim_z',
                       type=int,
                       default=wandb.config.dim_z if "dim_z" in wandb.config.keys() and isinstance(wandb.config.dim_z, int) else 500)

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
# argparser.add_argument('--weight_decay',
#                        type=float,
#                        default=0.0)
argparser.add_argument('--weight_decay',
                       type=float,
                       default=wandb.config.weight_decay if "weight_decay" in wandb.config.keys() and isinstance(wandb.config.weight_decay, float) else 0.0)

# argparser.add_argument('--max_grad_norm',
#                        type=float,
#                        default=1.0)
argparser.add_argument('--max_grad_norm',
                       type=float,
                       default=wandb.config.max_grad_norm if "max_grad_norm" in wandb.config.keys() and isinstance(wandb.config.max_grad_norm, float) else 1.0)

# argparser.add_argument('--lr',
#                        type=float,
#                        default=5e-4)

argparser.add_argument('--lr',
                       type=float,
                       default=wandb.config.lr if "lr" in wandb.config.keys() and isinstance(wandb.config.lr, float) else 5e-4)

argparser.add_argument('--disc_lr',
                       type=float,
                       default=wandb.config.disc_lr if "disc_lr" in wandb.config.keys() and isinstance(wandb.config.disc_lr, float) else 5e-5)

# argparser.add_argument("--temperature",
#                        type=float,
#                        default=0.1)

argparser.add_argument("--temperature",
                       type=float,
                       default=wandb.config.temperature if "temperature" in wandb.config.keys() and isinstance(wandb.config.temperature, float) else 0.1)

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
# argparser.add_argument('--threshold',
#                        type=float,
#                        default=None)
argparser.add_argument('--threshold',
                       type=float,
                       default=wandb.config.threshold if "threshold" in wandb.config.keys() and isinstance(wandb.config.threshold, float) else None)

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
    args.text_file_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/AIhub_train.csv'
    args.val_text_file_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/AIhub_test.csv'
    args.test_text_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/data/AIhub_test.csv'
    args.clf_ckpt_path = '/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer_jw/aihub_clf.pt'

elif args.dataset is None:
    assert args.text_file_path is not None
    assert args.val_text_file_path is not None
    assert args.test_text_path is not None

print('------------------------------------------------')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))
print('------------------------------------------------')
