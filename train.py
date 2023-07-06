'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from utils.tools import *
from utils.dataLoader import train_loader, eval_loader,my_collate_fn
from utils.ECAPAModel import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
parser.add_argument('--device',      type=str,   default='cuda:0',       help='Device training on ')
## Training Settings
parser.add_argument('--num_frames', type=int,   default=300,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=128,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="./data/train_df",     help='The path of the training list, eg:"/data08/VoxCeleb2/train_list.txt" in my case')
parser.add_argument('--eval_list',  type=str,   default="./data/valid_df",              help='The path of the evaluation list, eg:"/data08/VoxCeleb1/veri_test2.txt" in my case')
parser.add_argument('--eval_max_length', type=int,  default=120000,  help='the max length of evaluate audio')
parser.add_argument('--save_path',  type=str,   default="./exps",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the dialect encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=9,   help='Number of dialects')
parser.add_argument('--loss' ,   type=str, default="PairDistanceLoss", help='Target and loss function')
## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
train_data = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

def my_collate_fn(data, max_length=args.eval_max_length):

    lens = [x[0].shape[0] for x in data]
    max_len = max(lens)  
    max_length = min(max_len, max_length)

    features = torch.zeros(len(data), max_length)
    for i, length in enumerate(lens):
        features[i,:length] = data[i][0][:max_length]    
    labels = torch.tensor([x[1] for x in data])

    return features, labels

eval_data = eval_loader(**vars(args))
evalLoader = torch.utils.data.DataLoader(eval_data, batch_size = args.batch_size*2, shuffle = True, num_workers = args.n_cpu, drop_last = True, collate_fn=my_collate_fn)

## Search for the exist models
#modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
#modelfiles.sort()
modelfiles = []

## Only do evaluation, the initial_model is necessary
if args.eval == True:
    s = ECAPAModel(**vars(args))
    print("Model %s loaded from previous state!"%args.initial_model)
    s.load_parameters(args.initial_model)
    loss = s.eval_network(loader=evalLoader)
    print("EvalLoss %2.2f"%(loss))
    quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
    print("Model %s loaded from previous state!"%args.initial_model)
    s = ECAPAModel(**vars(args))
    s.load_parameters(args.initial_model)
    epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
    print("Model %s loaded from previous state!"%modelfiles[-1])
    epoch = 1
    s = ECAPAModel(**vars(args))
    s.load_parameters(modelfiles[-1])
    ## Otherwise, system will train from scratch
else:
    epoch = 1
    s = ECAPAModel(**vars(args))

eval_losses = []
score_file = open(args.score_save_path, "a+")

while(1):
    ## Training for one epoch
    loss, lr = s.train_network(epoch = epoch, loader = trainLoader)

    ## Evaluation every [test_step] epochs
    if epoch % args.test_step == 0:
        s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
        eval_losses.append(s.eval_network(loader=evalLoader))
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, EvalLoss %2.2f, minEvalLoss %2.2f"%(epoch,eval_losses[-1], min(eval_losses)))
        score_file.write("%d epoch, LR %f, LOSS %f, EvalLoss %2.2f, minEvalLoss %2.2f\n"%(epoch, lr, loss, eval_losses[-1], min(eval_losses)))
        score_file.flush()

    if epoch >= args.max_epoch:
        quit()

    epoch += 1
