import argparse
import random
import numpy as np
import torch
import os
from data_generator import Share_miniImagenet, MiniImagenet
from maml import MAML
from copy import deepcopy
from torch.distributions import Beta
parser = argparse.ArgumentParser(description='MLQA')
parser.add_argument('--datasource', default='miniimagenet', type=str,
                    help='miniimagenet')
parser.add_argument('--share', default=0, type=int,
                    help='wheather label sharing or not')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

## Training options
parser.add_argument('--metatrain_iterations', default=50000, type=int,
                    help='number of metatraining iterations.')
parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in maml')
parser.add_argument('--update_batch_size', default=1, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--num_filters', default=32, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--logdir', default='/home/fanjiangdong/workspace/MLQA-main/Miniimagenet/logdict', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='/data/data-home/liyuekeng', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=0, type=int, help='True to train, False to test.')
parser.add_argument('--ratio', default=0.2, type=float, help='train class ratio')
parser.add_argument('--test_set', default=1, type=int,
                    help='Set to true to test on the the test set, False for the validation set.')
parser.add_argument('--aug', default=True, action='store_true', help='use aug or not')
parser.add_argument('--mix', default=True, action='store_true', help='use mix or not')
parser.add_argument('--trial', default=0, type=int, help='trial')
parser.add_argument('--ablation', default=False, type=bool, help='ablation')
args = parser.parse_args()
print(args, flush=True)

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:2')
random.seed(1)
np.random.seed(2)

exp_string = f'MLQA.data_{args.datasource}.cls_{args.num_classes}.mbs_{args.meta_batch_size}.ubs_{args.update_batch_size}.metalr_{args.meta_lr}.innerlr_{args.update_lr}'

if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.mix:
    exp_string += '.mix'
if args.aug:
    exp_string += '.aug'
if args.ablation:
    exp_string += '.abl'
if args.trial > 0:
    exp_string += '.trial_{}'.format(args.trial)


print(exp_string, flush=True)

def train(args, maml, optimiser):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0
    best_acc, note_ci95 = 0.0, 0.0
    if args.share == 0:
        dataloader = MiniImagenet(args, 'train')
    elif args.share == 1:
        dataloader = Share_miniImagenet(args, 'train')

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')
    print("train begining!", flush=True)
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        if step > args.metatrain_iterations:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        task_losses = []
        task_acc = []

        for meta_batch in range(args.meta_batch_size):
            second_id = random.randint(0, args.meta_batch_size - 1) 
            # while second_id == meta_batch:
            #     second_id = random.randint(0, args.meta_batch_size - 1)
            if args.ablation:
                loss_val, acc_val = maml.forward_QA(x_spt[meta_batch], y_spt[meta_batch],
                                                            x_qry[meta_batch],
                                                            y_qry[meta_batch],
                                                            x_spt[second_id], y_spt[second_id],
                                                            x_qry[second_id],
                                                            y_qry[second_id])
            else:
                loss_val, acc_val = maml.forward_MLQA(x_spt[meta_batch], y_spt[meta_batch],
                                                                x_qry[meta_batch],
                                                                y_qry[meta_batch],
                                                                x_spt[second_id], y_spt[second_id],
                                                                x_qry[second_id],
                                                                y_qry[second_id])


            task_losses.append(loss_val)
            task_acc.append(acc_val)

        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_acc = torch.stack(task_acc).mean()

        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

        if step != 0 and step % Print_Iter == 0:
            print('iter: {}, loss_all: {}, acc: {}'.format(step, print_loss, print_acc), flush=True)

            print_loss, print_acc = 0.0, 0.0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            mean_acc, ci95 = test(args, maml, step)
            if mean_acc > best_acc:
                best_acc = mean_acc
                note_ci95 = ci95

                torch.save(maml.state_dict(),
                        '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))
    print('best_acc is {}, note_ci95 is {}'.format(best_acc, note_ci95), flush=True)


def test(args, maml, test_epoch):
    res_acc = []
    args.train = 0
    temp_meta_batch_size = args.meta_batch_size
    args.meta_batch_size = 1
    maml.num_updates = args.num_updates_test

    if args.share == 0:
        test_dataloader = MiniImagenet(args, 'test')
    elif args.share == 1:
        test_dataloader = Share_miniImagenet(args, 'test')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(test_dataloader):
        if step > args.num_test_task:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                    x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        copy_maml = deepcopy(maml)
        _, acc_val = copy_maml(x_spt, y_spt, x_qry, y_qry)
        del copy_maml
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)
    mean_acc = np.mean(res_acc)
    ci95 = 1.96 * np.std(res_acc) / np.sqrt(args.num_test_task * args.meta_batch_size)
    print('test_epoch is {}, acc is {}, ci95 is {}'.format(test_epoch, mean_acc, ci95), flush=True)
    
    args.meta_batch_size = temp_meta_batch_size
    args.train = 1
    maml.num_updates = args.num_updates
    return mean_acc, ci95

def main():
    maml = MAML(args).to(device)

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print(model_file, flush=True)
        maml.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(maml.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, maml, meta_optimiser)
    else:
        # model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        model_dir = "/home/fanjiangdong/workspace/MLQA-main/Medical_Image/logdict/MLQA.data_dermnet.cls_5.mbs_4.ubs_5.metalr_0.001.innerlr_0.01.hidden32.mix.aug"

        # 遍历文件夹下的所有文件
        for model_file in os.listdir(model_dir):
            # 拼接完整路径
            full_model_path = os.path.join(model_dir, model_file)
            
            # 检查是否为文件（确保不是子目录）
            if os.path.isfile(full_model_path):
                try:
                    print(f"Loading model from: {full_model_path}")
                    
                    # 加载模型
                    maml.load_state_dict(torch.load(full_model_path))
                    
                    # 测试模型
                    test(args, maml, args.test_epoch)
                    
                except Exception as e:
                    print(f"Error processing {full_model_path}: {e}")


if __name__ == '__main__':
    main()
