import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np
from numpy.lib.npyio import save
from numpy.random.mtrand import gamma
import torch.nn.functional as F
from torch import optim
import model_for_MI
import model_for_2b
import argparse
import torch
import deep_test27 
import CP_model
import new_model
import copy
from data2b import read_2b_data
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

def div(train,l,N=5):
    indt = np.arange(train.X.shape[0])
    np.random.seed(352465)
    np.random.shuffle(indt)
    n = len(train.X)
    step = n//N
    data_v = train.X[l*step:(l+1)*step]
    label_v = train.y[l*step:(l+1)*step] 
    data_t = np.concatenate([train.X[:l*step],train.X[(l+1)*step:]])
    label_t = np.concatenate([train.y[:l*step],train.y[(l+1)*step:]])

    return SignalAndTarget(data_t,label_t),SignalAndTarget(data_v,label_v)
    

def ica_pro(data,single_flag=False):
    ica = FastICA(n_components=22)
    

    result = ica.fit_transform(data)
    if single_flag :
        return result,ica
    else:
        return result
def data_arg(raw_data,D):
    data = raw_data.X
    label = raw_data.y
    
    indT = np.arange(data.shape[0])
    list_0 = np.where(label==0)[0]
    list_1 = np.where(label==1)[0]
    list_2 = np.where(label==2)[0]
    list_3 = np.where(label==3)[0]
    # assert data.shape[-1]%D == 0
    new_data = []
    new_data.append(data)
    new_label = []
    new_label.append(label)
    for l,d in enumerate([list_0,list_1,list_2,list_3]):
        data_temp = data[d]
        for k in range(1,data.shape[-1]//D):
            indT = np.arange(data_temp.shape[0])
            first,second = np.split(data_temp, [k*D], -1)
            np.random.seed(352465+k)
            np.random.shuffle(indT)
            first = first[indT]
            tmp = np.concatenate([first,second], -1)
            new_label.append(np.ones(d.shape[0])*l)
            new_data.append(tmp)
    new_data = np.concatenate(new_data,0)
    new_label = np.concatenate(new_label,0)
    indt = np.arange(new_data.shape[0])
    np.random.seed(352465)
    np.random.shuffle(indt)
    raw_data.X = new_data[indt]
    raw_data.y = new_label[indt]
    return raw_data

def run_exp(data_folder, subject_id, low_cut_hz, cuda,device):
    ival = [-500, 4000]
    max_epochs = 1600
    max_increase_epochs = 160
    batch_size = args.batch
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000
    device = torch.device("cuda:"+device)
    for i in range(1,10):
        # train_filename = "A{:02d}T.gdf".format(i)
        # test_filename = "A{:02d}E.gdf".format(i)
        # train_filepath = os.path.join(data_folder, train_filename)
        # test_filepath = os.path.join(data_folder, test_filename)
        # train_label_filepath = train_filepath.replace(".gdf", ".mat")
        # test_label_filepath = test_filepath.replace(".gdf", ".mat")

        # train_loader = BCICompetition4Set2A(
        #     train_filepath, labels_filename=train_label_filepath
        # )
        # test_loader = BCICompetition4Set2A(
        #     test_filepath, labels_filename=test_label_filepath
        # )
        # train_cnt = train_loader.load()
        # test_cnt = test_loader.load()

        # # Preprocessing

        # train_cnt = train_cnt.drop_channels(
        #     ["EOG-left", "EOG-central", "EOG-right"]
        # )
        # assert len(train_cnt.ch_names) == 22
        # # lets convert to millvolt for numerical stability of next operations
        # train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
        # train_cnt = mne_apply(
        #     lambda a: bandpass_cnt(
        #         a,
        #         low_cut_hz,
        #         high_cut_hz,
        #         train_cnt.info["sfreq"],
        #         filt_order=3,
        #         axis=1,
        #     ),
        #     train_cnt,
        # )
    
        # train_cnt = mne_apply(
        #     lambda a: exponential_running_standardize(
        #         a.T,
        #         factor_new=factor_new,
        #         init_block_size=init_block_size,
        #         eps=1e-4,
        #     ).T,
        #     train_cnt,
        # )

        # test_cnt = test_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
        # assert len(test_cnt.ch_names) == 22
        # test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
        # test_cnt = mne_apply(
        #     lambda a: bandpass_cnt(
        #         a,
        #         low_cut_hz,
        #         high_cut_hz,
        #         test_cnt.info["sfreq"],
        #         filt_order=3,
        #         axis=1,
        #     ),
        #     test_cnt,
        # )
        
        # test_cnt = mne_apply(
        #     lambda a: exponential_running_standardize(
        #         a.T,
        #         factor_new=factor_new,
        #         init_block_size=init_block_size,
        #         eps=1e-4,
        #     ).T,
        #     test_cnt,
        # )

        # marker_def = OrderedDict(
        #     [
        #         ("Left Hand", [1]),
        #         ("Right Hand", [2]),
        #         ("Foot", [3]),
        #         ("Tongue", [4]),
        #     ]
        # )

        # train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
        # test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

        # # train_set, valid_set = split_into_two_sets(
        # #     train_set, first_set_fraction=1 - valid_set_fraction
        # # )

        # #4hz for shallow
        # train_cnt = train_loader.load()
        # test_cnt = test_loader.load()

        # # Preprocessing

        # train_cnt = train_cnt.drop_channels(
        #     ["EOG-left", "EOG-central", "EOG-right"]
        # )
        # assert len(train_cnt.ch_names) == 22
        # # lets convert to millvolt for numerical stability of next operations
        # train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
        # train_cnt = mne_apply(
        #     lambda a: bandpass_cnt(
        #         a,
        #         4,
        #         high_cut_hz,
        #         train_cnt.info["sfreq"],
        #         filt_order=3,
        #         axis=1,
        #     ),
        #     train_cnt,
        # )
    
        # train_cnt = mne_apply(
        #     lambda a: exponential_running_standardize(
        #         a.T,
        #         factor_new=factor_new,
        #         init_block_size=init_block_size,
        #         eps=1e-4,
        #     ).T,
        #     train_cnt,
        # )

        # test_cnt = test_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
        # assert len(test_cnt.ch_names) == 22
        # test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
        # test_cnt = mne_apply(
        #     lambda a: bandpass_cnt(
        #         a,
        #         4,
        #         high_cut_hz,
        #         test_cnt.info["sfreq"],
        #         filt_order=3,
        #         axis=1,
        #     ),
        #     test_cnt,
        # )
        
        
        # test_cnt = mne_apply(
        #     lambda a: exponential_running_standardize(
        #         a.T,
        #         factor_new=factor_new,
        #         init_block_size=init_block_size,
        #         eps=1e-4,
        #     ).T,
        #     test_cnt,
        # )

        # marker_def = OrderedDict(
        #     [
        #         ("Left Hand", [1]),
        #         ("Right Hand", [2]),
        #         ("Foot", [3]),
        #         ("Tongue", [4]),
        #     ]
        # )
        
    
        # train_set_4 = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
        # test_set_4 = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
        
        

        train_set = read_2b_data(patient=i,Type = 'train',ival=[-125,1000],low_cut_hz=0)
        test_set = read_2b_data(patient=i,Type = 'test',ival=[-125,1000],low_cut_hz=0)
        train_set_4 = read_2b_data(patient=i,Type = 'train',ival=[-125,1000])
        test_set_4 = read_2b_data(patient=i,Type = 'test',ival=[-125,1000])
        

        
        test_data = np.concatenate([test_set.X[:,:,:,np.newaxis],test_set_4.X[:,:,:,np.newaxis]],-1)
        test_set.X = test_data
        train_data = np.concatenate([train_set.X[:,:,:,np.newaxis],train_set_4.X[:,:,:,np.newaxis]],-1)
        train_set.X = train_data
        train,valid_set = div(train_set,0)
        test,valid_set_test = div(test_set,0)

        train = data_arg(train,args.aug)
        test =data_arg(test,args.aug)
        valid_set = data_arg(valid_set,args.aug)
        valid_set_test = data_arg(valid_set_test,args.aug)

        
        if i == subject_id and i != 1:
            target_test_set = copy.deepcopy(test_set)
            target_set = copy.deepcopy(data_arg(train_set,args.aug))
            source_set_train.X = np.concatenate((source_set_train.X,train.X),0)
            source_set_train.y = np.concatenate((source_set_train.y,train.y),0)
            source_set_val.X = np.concatenate((source_set_val.X,valid_set.X),0)
            source_set_val.y = np.concatenate((source_set_val.y,valid_set.y),0)
        elif i ==1 and i == subject_id:
            target_test_set = copy.deepcopy(test_set)
            target_set = copy.deepcopy(data_arg(train_set,args.aug))
            source_set_train = copy.deepcopy(train)
            source_set_val = copy.deepcopy(valid_set)

        elif i == 1 or (subject_id == 1 and i == 2):
            source_set_train = copy.deepcopy(train)
            source_set_val = copy.deepcopy(valid_set)
            source_set_train.X = np.concatenate((source_set_train.X,test.X),0)
            source_set_train.y = np.concatenate((source_set_train.y,test.y),0)
            source_set_val.X = np.concatenate((source_set_val.X,valid_set_test.X),0)
            source_set_val.y = np.concatenate((source_set_val.y,valid_set_test.y),0)
        else:
            source_set_train.X = np.concatenate((source_set_train.X,train.X,test.X),0)
            source_set_train.y = np.concatenate((source_set_train.y,train.y,test.y),0)
            source_set_val.X = np.concatenate((source_set_val.X,valid_set.X,valid_set_test.X),0)
            source_set_val.y = np.concatenate((source_set_val.y,valid_set.y,valid_set_test.y),0)  
    set_random_seeds(seed=20190706, cuda=cuda)

    # model= shallow_test.shallow_deep()
    model = model_for_2b.shallow_deep()

    if cuda:
        model.to(device)
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # optimizer = optim.SGD(model.parameters(),lr=0.002,momentum=1)
    # optimizer = optim.Adadelta(model.parameters(),lr=0.002)
    # optimizer = optim.RMSprop(model.parameters())
    # LR = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98)
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
    model.feature_flag=True
    exp = Experiment(
        subject_id,
        model,
        source_set_train,
        source_set_val,
        target_test_set,
        iterator=iterator,
        loss_function=F.nll_loss,
        optimizer=optimizer,
        model_constraint=model_constraint,
        monitors=monitors,
        stop_criterion=stop_criterion,
        remember_best_column="valid_misclass",
        run_after_early_stop=True,
        cuda=cuda,
        target_set = target_set,
        transfer_learning = True,
        device = device,
        save_flag = True 
    )
    exp.run()
    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))
    return exp
    
parser = argparse.ArgumentParser(description='命令行中传入作为测试集的发作次数')   
parser.add_argument('--patient', type=int ,help='选择的病人',default = 99)
parser.add_argument('--PrintOrLog', type=str ,help='结果是打印还是存到本地log文件',default = 'print')
parser.add_argument('--gpu', type=str ,help='选择GPU卡号',default = '0')
parser.add_argument('--batch', type=int ,help='batch-size',default = 256)
parser.add_argument('--save', type=bool ,help='是否保存网络参数',default = False)
parser.add_argument('--aug', type=int ,help='数据增强D ',default = 50)
args = parser.parse_args() 



if __name__ == "__main__":
    if args.PrintOrLog == 'log':

        logging.basicConfig(
            format="%(asctime)s %(levelname)s : %(message)s",
            level=logging.DEBUG,
            filename='log/model_deep_shallow.log',
        )
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s : %(message)s",
            level=logging.DEBUG,
            # stream=sys.stdout,
            
        )
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = "/data/xys/MI/raw_data/"
    
    low_cut_hz = 0 # 0 or 4
    cuda = True
    for i in range(9):
        if args.patient == 99:
            subject_id = i+1
        else:
            subject_id = args.patient  # 1-9

        exp = run_exp(data_folder, subject_id, low_cut_hz,  cuda,device = args.gpu)
        
        # net = exp.model.to(torch.device('cpu'))
        # torch.save(net.state_dict(),'test_net_shallow.pt')
        if args.patient != 99:
            break
    qqmail('net:shallow_deep patient:{}'.format(args.patient))
    
