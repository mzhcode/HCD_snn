import numpy as np
import time
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from thop import clever_format, profile

# import model
import test
from data_pre import load_dataset, sampling, get_mask_onehot, Grammar, zeropad_to_max_len, Data_Generator, get_position
from sampling import get_coordinates_labels, get_train_test, get_train_val_test
from dataloader import simple_data_generator
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from dataloader import simple_data_generator_test

import warnings

warnings.filterwarnings('ignore')

import argparse
# from thop import profile,clever_format
seeds = [1337, 1338, 1339, 1340, 1341]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
selection_rules = ["rect 3"]
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=150, help='n_epochs')
    parser.add_argument("--max_depth", type=int, default=1, help="max_depth")
    parser.add_argument('--batch_size', type=int, default=128, help="num_batches")
    parser.add_argument("--num_head", type=int, default=10)
    parser.add_argument("--drop_rate", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--log_every_n_samples", type=int, default=5)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="models")
    parser.add_argument("--test_size",type=float, default=0.95)
    parser.add_argument("--val_size", type=float, default=0.02)
    parser.add_argument("--start_learning_rate", type=float, default=4e-4)
    parser.add_argument("--dataset", type=str, default="RV")
    parser.add_argument("--prembed", type=bool, default=True)
    parser.add_argument("--prembed_dim", type=int, default=30)
    parser.add_argument("--data_path", type=str, default="/home/think/spikehcd_test/farmland/")
    parser.add_argument("--repeat_term", type=int, default=1)
    parser.add_argument("--is_valid", type=bool, default=False)
    parser.add_argument("--limited_num", type=int, default=50)
    parser.add_argument("--num_hidden", type=int, default=30)
    parser.add_argument("--max_len", type=int,default=9)
    args = parser.parse_args()
    return args

def GT_To_One_Hot(gt):
    GT_One_Hot = []
    for i in range(gt.shape[0]):
        if gt[i] != 0 :
            temp = [0, 1]
        else:
            temp = [1, 0]
        GT_One_Hot.append(temp)
    return GT_One_Hot

global Dataset

Dataset = "FM"  # RV, FM, USA

sampling_mode = "random"
margin = 4

X1, X2, y, dataset_name = load_dataset(Dataset)
xshape = X1.shape[1:]
arg = get_args()
height, width, bands = X1.shape
OA_record = []
kappa_record = []
alltime_record = []

def cal_kappa(pred, truth):

    pred = pred.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()
    cal_label = 1 - truth
    cal_pred = 1 - pred
    tp = np.sum(np.logical_and(cal_pred, cal_label))
    tn = np.sum(np.logical_not(np.logical_or(cal_pred, cal_label)))
    fp = np.sum(np.logical_and(np.logical_xor(cal_pred, cal_label), cal_pred))
    fn = np.sum(np.logical_and(np.logical_xor(cal_pred, cal_label), cal_label))

    return tp, tn, fp, fn

for repterm in range(arg.repeat_term):

    print("start")
    print(seeds[repterm])
    np.random.seed(seeds[repterm])
    coords, labels = get_coordinates_labels(y)

    train_coords, train_labels, test_coords, test_labels, val_coords, val_labels = get_train_test(data=coords, data_labels=labels, val_size=arg.val_size,
                                                                          test_size=arg.test_size)

    train_coords = train_coords + margin
    test_coords = test_coords + margin
    val_coords = val_coords +margin

    X_train1, X_train2 = Grammar(X1, X2, train_coords, method="rect 3")
    X_val1, X_val2 = Grammar(X1, X2, val_coords, method = "rect 3")

    y_train = train_labels
    y_test = test_labels
    y_val = val_labels
    X_train_shape = X_train1.shape
    X_val_shape = X_val1.shape
    if len(X_train_shape) == 4:
        X_train1 = np.reshape(X_train1, [X_train_shape[0], X_train_shape[1] * X_train_shape[2],
                                       X_train_shape[3]])
        X_train2 = np.reshape(X_train2, [X_train_shape[0], X_train_shape[1] * X_train_shape[2],
                                         X_train_shape[3]])

        X_val1 = np.reshape(X_val1, [X_val_shape[0], X_val_shape[1] * X_val_shape[2], X_val_shape[3]])
        X_val2 = np.reshape(X_val2, [X_val_shape[0], X_val_shape[1] * X_val_shape[2], X_val_shape[3]])

    X_train1 = zeropad_to_max_len(X_train1, max_len=arg.max_len)
    X_train2 = zeropad_to_max_len(X_train2, max_len=arg.max_len)
    X_val1 = zeropad_to_max_len(X_val1, max_len=arg.max_len)
    X_val2 = zeropad_to_max_len(X_val2, max_len=arg.max_len)

    X_test1, X_test2 = Grammar(X1, X2, test_coords, method="rect 3")
    X_test_shape = X_test1.shape
    y = 2 - y
    pred = y
    realcoords = test_coords - 4
    if len(X_test_shape) == 4:
        X_test1 = np.reshape(X_test1, [X_test_shape[0], X_test_shape[1] * X_test_shape[2], X_test_shape[3]])
        X_test2 = np.reshape(X_test2, [X_test_shape[0], X_test_shape[1] * X_test_shape[2], X_test_shape[3]])
    X_test1 = zeropad_to_max_len(X_test1, max_len=arg.max_len)
    X_test2 = zeropad_to_max_len(X_test2, max_len=arg.max_len)

    for i in range(2):
        print("num train and test in class %d is %d / %d" % (
        i, (y_train == i).sum(), (y_test == i).sum()))

    import model_farmland
    net = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.2,
        drop_block_rate=None,
        img_size_h=3, img_size_w=3,
        patch_size=3, embed_dims=128, num_heads=4, mlp_ratios=4,
        in_channels=155, num_classes=2, qkv_bias=False,
        depths=3, sr_ratios=1,
    )

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=arg.start_learning_rate)
    best_loss = 99999
    best_oa = 0

    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)

    for i in range(arg.n_epochs+1):
        net.train()
        train_generator = simple_data_generator(X_train1, X_train2, y_train, batch_size=arg.batch_size, shuffle=True)
        val_generator = simple_data_generator(X_val1, X_val2, y_val, batch_size=arg.batch_size, shuffle=True)
        test_generator = simple_data_generator_test(X_test1, X_test2, y_test, realcoords, batch_size=arg.batch_size,
                                                    shuffle=True)
        sample_train_num = np.shape(y_train)
        sample_test_num = np.shape(y_test)
        sample_val_num = np.shape(y_val)
        total_train_eq = 0
        total_train = 0
        total_train_loss = 0
        train_total_batch = 0
        val_total_batch = 0
        val_total_batch = 0
        total_val_eq = 0
        total_val_loss = 0

        total_test_eq = 0
        total_tp = total_tn = total_fp = total_fn = 0

        for ind, (X_batch1, X_batch2, y_batch) in enumerate(train_generator):
            X_batch1 = torch.from_numpy(X_batch1.astype(np.float32)).to(device)
            X_batch2 = torch.from_numpy(X_batch2.astype(np.float32)).to(device)
            train_batch_num = X_batch1.shape[0]
            train_batch_num = train_batch_num
            train_total_batch = train_total_batch + train_batch_num
            y_batch = torch.from_numpy(y_batch.astype(np.float32)).to(device)
            optimizer.zero_grad()
            output = net(X_batch1, X_batch2)
            loss = loss_fn(output, y_batch.long())
            loss = torch.sum(loss)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_cpu = loss.cpu().detach().numpy()
            total_train_loss = loss_cpu + total_train_loss
            logits = torch.argmax(output, dim=1)
            equal_num = torch.eq(logits, y_batch)
            equal_num = torch.sum(equal_num)
            equal_num = equal_num.cpu().numpy()
            total_train_eq = total_train_eq + equal_num

        net.eval()
        for ind, (X_batch1, X_batch2, y_batch) in enumerate(val_generator):
            with torch.no_grad():
                X_batch1 = torch.from_numpy(X_batch1.astype(np.float32)).to(device)
                X_batch2 = torch.from_numpy(X_batch2.astype(np.float32)).to(device)
                val_batch_num = X_batch1.shape[0]
                val_batch_num = val_batch_num
                val_total_batch = val_total_batch + val_batch_num
                y_batch = torch.from_numpy(y_batch.astype(np.float32)).to(device)
                output = net(X_batch1, X_batch2)
                loss = loss_fn(output, y_batch.long())
                loss = torch.sum(loss)
                loss_cpu = loss.cpu().detach().numpy()
                total_val_loss = loss_cpu + total_val_loss
                logits = torch.argmax(output, dim=1)
                equal_num = torch.eq(logits, y_batch)
                equal_num = torch.sum(equal_num)
                equal_num = equal_num.cpu().numpy()
                total_val_eq = total_val_eq + equal_num

        train_oa = total_train_eq / train_total_batch
        val_oa = total_val_eq / val_total_batch


        net.eval()
        for ind, (X_batch1, X_batch2, y_batch, order) in enumerate(test_generator):
            with torch.no_grad():
                X_batch1 = torch.from_numpy(X_batch1.astype(np.float32)).to(device)
                X_batch2 = torch.from_numpy(X_batch2.astype(np.float32)).to(device)
                y_batch = torch.from_numpy(y_batch.astype(np.float32)).to(device)
                output = net(X_batch1, X_batch2)
                logits = torch.argmax(output, dim=1)
                TP, TN, FP, FN = cal_kappa(logits, y_batch)
                total_tp = total_tp + TP
                total_tn = total_tn + TN
                total_fp = total_fp + FP
                total_fn = total_fn + FN
                locpu = logits.detach().cpu()
                loo = 1 - locpu
                for j in range(len(loo)):
                    a = order[j][0]
                    b = order[j][1]
                    if locpu[j] == 1 and y_batch[j] == 1:
                        pred[a][b] = 0  # True Positive (TP)
                    elif locpu[j] == 0 and y_batch[j] == 0:
                        pred[a][b] = 1  # True Negative (TN)
                    elif locpu[j] == 1 and y_batch[j] == 0:
                        pred[a][b] = 3  # False Positive (FP)
                    elif locpu[j] == 0 and y_batch[j] == 1:
                        pred[a][b] = 2  # False Negative (FN)

        total_num = total_tp + total_tn + total_fp + total_fn
        total_num2 = total_num
        test_OA = (total_tp + total_tn) / total_num
        suanzi1 = (total_tp + total_fp) / total_num * (total_tp + total_fn) / total_num
        suanzi2 = (total_tn + total_fp) / total_num * (total_tn + total_fn) / total_num
        PRE = suanzi1 + suanzi2
        test_kappa = (test_OA - PRE) / (1 - PRE)

        print(i, "    train loss: ", total_train_loss, "    train oa: ", train_oa, "val loss", total_val_loss, "    val oa: ", val_oa, "test_OA", test_OA, "test_kappa", test_kappa)
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            torch.save(net.state_dict(), "model/best_model_FM_SpikeHCD.pt")
            print('save model...')
            torch.cuda.empty_cache()

        if best_oa < test_OA:
            best_oa = test_OA
            print(best_oa)



