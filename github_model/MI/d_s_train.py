from data2b import read_2b_data
import logging
import os.path
from tarfile import TarError
import time
from collections import OrderedDict
import sys
import csv
import numpy as np
from numpy.random.mtrand import gamma
import torch.nn.functional as F
from torch import optim

import model_for_MI
import argparse
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.autograd import Variable
import deep_test27
import CP_model
import new_model
import copy
import calculate
from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
import shallow_test
import model_for_2b
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from sklearn.decomposition import FastICA
from braindecode.datautil.signal_target import SignalAndTarget
import mne
from finish import send_mail_to_phone as qqmail
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
log = logging.getLogger(__name__)


def div(train, l, N=5):
    indt = np.arange(train.X.shape[0])
    np.random.seed(352465)
    np.random.shuffle(indt)
    n = len(train.X)
    step = n//N
    data_v = train.X[l*step:(l+1)*step]
    label_v = train.y[l*step:(l+1)*step]
    data_t = np.concatenate([train.X[:l*step], train.X[(l+1)*step:]])
    label_t = np.concatenate([train.y[:l*step], train.y[(l+1)*step:]])

    return SignalAndTarget(data_t, label_t), SignalAndTarget(data_v, label_v)


def ica_pro(data, single_flag=False):
    ica = FastICA(n_components=22)

    result = ica.fit_transform(data)
    if single_flag:
        return result, ica
    else:
        return result


def data_arg(raw_data, D):
    data = raw_data.X
    label = raw_data.y

    indT = np.arange(data.shape[0])
    list_0 = np.where(label == 0)[0]
    list_1 = np.where(label == 1)[0]
    list_2 = np.where(label == 2)[0]
    list_3 = np.where(label == 3)[0]
    # assert data.shape[-1]%D == 0
    new_data = []
    new_data.append(data)
    new_label = []
    new_label.append(label)
    for l, d in enumerate([list_0, list_1, list_2, list_3]):
        data_temp = data[d]
        for k in range(1, data.shape[-1]//D):
            indT = np.arange(data_temp.shape[0])
            first, second = np.split(data_temp, [k*D], -1)
            np.random.seed(352465+k)
            np.random.shuffle(indT)
            first = first[indT]
            tmp = np.concatenate([first, second], -1)
            new_label.append(np.ones(d.shape[0])*l)
            new_data.append(tmp)
    new_data = np.concatenate(new_data, 0)
    new_label = np.concatenate(new_label, 0)
    indt = np.arange(new_data.shape[0])
    np.random.seed(352465)
    np.random.shuffle(indt)
    raw_data.X = new_data[indt]
    raw_data.y = new_label[indt]
    return raw_data


class input_data(Dataset):
    def __init__(self, x, y):
        self.x_data = torch.from_numpy(x).unsqueeze(1)
        self.y_data = torch.from_numpy(y).squeeze()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.size()[0]


def run_exp(data_folder, subject_id, cuda, device):
    device = torch.device("cuda:"+device)
    ival = [-500, 4000]
    max_epochs = 1600
    max_increase_epochs = 160
    batch_size = args.batch
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000
    if(args.dataset == 'a'):
        train_filename = "A{:02d}T.gdf".format(subject_id)
        test_filename = "A{:02d}E.gdf".format(subject_id)
        train_filepath = os.path.join(data_folder, train_filename)
        test_filepath = os.path.join(data_folder, test_filename)
        train_label_filepath = train_filepath.replace(".gdf", ".mat")
        test_label_filepath = test_filepath.replace(".gdf", ".mat")

        train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath
        )
        test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath
        )
        train_cnt = train_loader.load()
        test_cnt = test_loader.load()
        train_cnt = train_cnt.drop_channels(
            ["EOG-left", "EOG-central", "EOG-right"]
        )
        test_cnt = test_cnt.drop_channels(
            ["EOG-left", "EOG-central", "EOG-right"]
        )
        assert len(train_cnt.ch_names) == 22
        assert len(test_cnt.ch_names) == 22
        test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
        train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)

        train_set_0 = mne_apply(
            lambda a: bandpass_cnt(
                a,
                0,
                high_cut_hz,
                train_cnt.info["sfreq"],
                filt_order=3,
                axis=1,
            ),
            train_cnt,
        )
        train_set_4 = mne_apply(
            lambda a: bandpass_cnt(
                a,
                0,
                high_cut_hz,
                train_cnt.info["sfreq"],
                filt_order=3,
                axis=1,
            ),
            train_cnt,
        )
        test_set_0 = mne_apply(
            lambda a: bandpass_cnt(
                a,
                4,
                high_cut_hz,
                train_cnt.info["sfreq"],
                filt_order=3,
                axis=1,
            ),
            test_cnt,
        )
        test_set_4 = mne_apply(
            lambda a: bandpass_cnt(
                a,
                4,
                high_cut_hz,
                train_cnt.info["sfreq"],
                filt_order=3,
                axis=1,
            ),
            test_cnt,
        )
        train_set_0 = mne_apply(
            lambda a: exponential_running_standardize(
                a.T,
                factor_new=factor_new,
                init_block_size=init_block_size,
                eps=1e-4,
            ).T,
            train_set_0,
        )
        test_set_0 = mne_apply(
            lambda a: exponential_running_standardize(
                a.T,
                factor_new=factor_new,
                init_block_size=init_block_size,
                eps=1e-4,
            ).T,
            test_set_0,
        )
        train_set_4 = mne_apply(
            lambda a: exponential_running_standardize(
                a.T,
                factor_new=factor_new,
                init_block_size=init_block_size,
                eps=1e-4,
            ).T,
            train_set_4,
        )
        test_set_4 = mne_apply(
            lambda a: exponential_running_standardize(
                a.T,
                factor_new=factor_new,
                init_block_size=init_block_size,
                eps=1e-4,
            ).T,
            test_set_4,
        )
        marker_def = OrderedDict(
            [
                ("Left Hand", [1]),
                ("Right Hand", [2]),
                ("Foot", [3]),
                ("Tongue", [4]),
            ]
        )
        train_set_0 = create_signal_target_from_raw_mne(train_set_0, marker_def, ival)
        test_set_0 = create_signal_target_from_raw_mne(test_set_0, marker_def, ival)
        train_set_4 = create_signal_target_from_raw_mne(train_set_4, marker_def, ival)
        test_set_4 = create_signal_target_from_raw_mne(test_set_4, marker_def, ival)
        model_agent = shallow_test.shallow_deep()
        path = './result/tl/new/'+ str(subject_id) + '.pt'
    else:
        train_set_0= read_2b_data(patient=subject_id,Type = 'train',ival=[-125,1000],low_cut_hz=0)
        test_set_0 = read_2b_data(patient=subject_id,Type = 'test',ival=[-125,1000],low_cut_hz=0)
        train_set_4 = read_2b_data(patient=subject_id,Type = 'train',ival=[-125,1000])
        test_set_4 = read_2b_data(patient=subject_id,Type = 'test',ival=[-125,1000])
        model_agent = model_for_2b.shallow_deep()
        path = './result/'+ str(subject_id) + '/best.pt'

    eval_data = input_data(np.concatenate([train_set_0.X[:,:,:,np.newaxis],train_set_4.X[:,:,:,np.newaxis]],-1),train_set_0.y)
    train,valid_set = div(train_set_0,4)
    train_4,valid_set_4 = div(train_set_4,4)
    train = data_arg(train,args.aug)
    valid_set = data_arg(valid_set,args.aug)
    train_4 = data_arg(train_4,args.aug)
    valid_set_4 = data_arg(valid_set_4,args.aug)
    train_data = np.concatenate([train.X[:,:,:,np.newaxis],train_4.X[:,:,:,np.newaxis]],-1)
    valid_data = np.concatenate([valid_set.X[:,:,:,np.newaxis],valid_set_4.X[:,:,:,np.newaxis]],-1)

    
    train.X = train_data
    valid_set.X = valid_data
    
    test_data = np.concatenate([test_set_0.X[:,:,:,np.newaxis],test_set_4.X[:,:,:,np.newaxis]],-1)
    test_set_0.X = test_data

    input_dict =torch.load(path)
    model_agent.load_state_dict(input_dict)

    for l in range(2):
        set_random_seeds(seed=20190706, cuda=cuda)  
        if(args.dataset == 'a'):
            model = shallow_test.shallow_deep()
        else:
            model = model_for_2b.shallow_deep_t()
        model.load_state_dict(input_dict)
        raw_dict = model.state_dict()
        if cuda:
            model.to(device)
        log.info("Model: \n{:s}".format(str(model)))
        if l == 0:
            model_dict = model_agent.conv_time_d.state_dict()
            input_state_dict = {'conv_time_d.'+k:v for k,v in model_dict.items()}
            # raw_dict.update(input_state_dict)#更新
            for name,para in model.conv_time_d.named_parameters():
                para.requires_grad = False
            
            
            model_dict = model_agent.conv_spat_d.state_dict()
            input_state_dict = {'conv_spat_d.'+k:v for k,v in model_dict.items()}
            # raw_dict.update(input_state_dict)#更新
            for name,para in model.conv_spat_d.named_parameters():
                para.requires_grad = False


            model_dict = model_agent.bn_d.state_dict()
            input_state_dict = {'bn_d.'+k:v for k,v in model_dict.items()}
            # raw_dict.update(input_state_dict)#更新
            for name,para in model.bn_d.named_parameters():
                para.requires_grad = False

            model_dict = model_agent.deep.state_dict()
            input_state_dict = {'deep.'+k:v for k,v in model_dict.items()}
            # raw_dict.update(input_state_dict)#更新
            for name,para in model.deep.named_parameters():
                para.requires_grad = False
        else:
            model_dict = model_agent.conv_time_s.state_dict()
            input_state_dict = {'conv_time_s.'+k:v for k,v in model_dict.items()}
            # raw_dict.update(input_state_dict)#更新
            for name,para in model.conv_time_s.named_parameters():
                para.requires_grad = False
            
            
            model_dict = model_agent.conv_spat_s.state_dict()
            input_state_dict = {'conv_spat_s.'+k:v for k,v in model_dict.items()}
            # raw_dict.update(input_state_dict)#更新
            for name,para in model.conv_spat_s.named_parameters():
                para.requires_grad = False


            model_dict = model_agent.bn_s.state_dict()
            input_state_dict = {'bn_s.'+k:v for k,v in model_dict.items()}
            # raw_dict.update(input_state_dict)#更新
            for name,para in model.bn_s.named_parameters():
                para.requires_grad = False
        # model.load_state_dict(raw_dict)
        try:
            model.feature_flag = False
        except AttributeError:
            print('no feature_flag')
            return
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))   
        LR = CosineAnnealingWarmRestarts(optimizer,T_0=150,T_mult=1)
        iterator = BalancedBatchSizeIterator(batch_size=batch_size)
        stop_criterion = Or(
            [
                MaxEpochs(max_epochs),
                NoDecrease("valid_misclass", max_increase_epochs),
            ]
        )
        monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
        model_constraint = MaxNormDefaultConstraint()
        exp = Experiment(
                subject_id,
                model,
                train,
                valid_set,
                test_set_0,
                iterator=iterator,
                loss_function=F.nll_loss,
                optimizer=optimizer,
                model_constraint=model_constraint,
                monitors=monitors,
                stop_criterion=stop_criterion,
                remember_best_column="valid_misclass",
                run_after_early_stop=True,
                cuda=cuda,
                device = device,
                lr_decay = LR,
                save_flag = args.save,
                feature_loss=False
            )
        exp.run()
        acc,kappa,f1,auc,con = calculate.calculate(exp.model,test_set_0,device)
        with open('/data/xys/MI/feature_net/'+args.dataset+'/result.csv', 'a') as csvfile:
            writer = csv.writer(csvfile,delimiter = ',')
            writer.writerow([subject_id,l,acc,kappa,f1,auc,con])   
        print("acc:{} kappa:{} f1:{} auc:{} ".format(acc,kappa,f1,auc))
        # 特诊可视化
        model = exp.model
        model.feature_flag=True  
        model.eval()  
        eval_indices = list(range(eval_data.__len__()))
            # target_len = target_data.__len__() 
        eval_train = Data.Subset(eval_data , eval_indices)  
        eval_loader = Data.DataLoader(eval_train,
                                    batch_size = batch_size,
                                    shuffle = True)
        save_deep  =[]
        save_shallow = []
        for batch_idx, (data, target) in enumerate(eval_loader):
            data, target = Variable(data).to(device), Variable(target).to(device)
            data = data.type(torch.cuda.FloatTensor).squeeze()            
            _,deep_feature,shallow_feature = model(data)
            save_deep.append(deep_feature.cpu().detach().numpy())
            save_shallow.append(shallow_feature.cpu().detach().numpy())
        if(len(save_deep)>1):
            save_deep= np.concatenate(save_deep)
            save_shallow = np.concatenate(save_shallow)
        else:
            save_deep = save_deep[0]
            save_shallow = save_shallow[0]
        save_path = '/data/xys/MI/feature_net/'+args.dataset+'/'
        if l == 0:
            np.save(save_path+str(subject_id)+"_deepfeatureD.npy",save_deep)
            np.save(save_path+str(subject_id)+"_shallowfeatureD.npy",save_shallow)
            np.save(save_path+str(subject_id)+"_label.npy",train_set_0.y)
        elif l == 1:
            np.save(save_path+str(subject_id)+"_deepfeatureS.npy",save_deep)
            np.save(save_path+str(subject_id)+"_shallowfeatureS.npy",save_shallow)
    return exp
parser = argparse.ArgumentParser(description='命令行中传入作为测试集的发作次数')
parser.add_argument('--patient', type=int, help='选择的病人', default=99)
parser.add_argument('--PrintOrLog', type=str,
                    help='结果是打印还是存到本地log文件', default='print')
parser.add_argument('--gpu', type=str, help='选择GPU卡号', default='0')
parser.add_argument('--batch', type=int, help='batch-size', default=256)
parser.add_argument('--save', type=bool, help='是否保存网络参数', default=False)
parser.add_argument('--aug', type=int, help='数据增强D ', default=50)
parser.add_argument('--dataset', type=str, help='数据集IIa或者IIb ', default='a')
args = parser.parse_args()


if __name__ == "__main__":
    out = []
    if args.PrintOrLog == 'log':

        logging.basicConfig(
            format="%(asctime)s %(levelname)s : %(message)s",
            level=logging.DEBUG,
            filename='/feature_net/'+args.dataset+'.log',
        )
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s : %(message)s",
            level=logging.DEBUG,
        )

    data_folder = "/data/xys/MI/raw_data/"

    low_cut_hz = 0  # 0 or 4
    cuda = True
    with open('/data/xys/MI/feature_net/'+args.dataset+'/result.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'l', 'acc', 'kappa',
                        'F1', 'AUC', 'confusion'])
    for i in range(9):
        if args.patient == 99:
            subject_id = i+1
        else:
            subject_id = args.patient  # 1-9
        exp = run_exp(data_folder, subject_id,
                       cuda, device=args.gpu)
        if args.patient != 99:
            break
