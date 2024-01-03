# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:54:06 2021

@author: XYS
"""
import numpy as np
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import argparse
from model import SeizureNet
from sklearn.metrics import roc_curve,auc


data_total = {'chb1':7,
              'chb2':3,
              'chb3':7,
              # 'chb4':4,
              'chb5':5,
              'chb6':7,
              'chb7':3,
              'chb8':5,
              'chb9':4,
              'chb10':7,
              'chb11':3,
              'chb14':8,
              'chb16':7,
              'chb17':3,
              'chb18':5,
              'chb19':3,
              'chb20':8,
              'chb21':4,
              'chb22':3,
              'chb23':7
              
               } 
data_check = {'chb1':[0,2,6],
              'chb2':[0,2],
              'chb3':[0,4],
              'chb5':[0,1,2,4],
              'chb6':[1,2,4,5,6],
              'chb7':[0,1,2],
              'chb8':[0,1,2,4],
              'chb9':[0,1,3],
              'chb10':[0,1,2,3,4,5,6],
              'chb11':[0,1,2],
              'chb14':[0,4,5,7],
              'chb16':[0,2],
              'chb17':[0,2],
              'chb18':[0,1,3,4],
              'chb19':[0,1,2],
              'chb20':[0,7],
              'chb21':[0,2],
              'chb22':[0,1,2],
              'chb23':[0,3]

    }



chb_path = {  'chb1':'/home/xys/test1/result/chb1/net/0_total/count7epo31.pt',
              'chb2':'/home/xys/test1/result/chb2/net/0_total/count5epo158.pt',
              'chb3':'/home/xys/test1/result/chb3/net/0_total/count7epo72.pt',
              'chb4':'/home/xys/test1/result/chb4/net/0_total/count2epo50.pt',
              'chb5':'/home/xys/test1/result/chb5/net/0_total/best_epo.pt',
              'chb6': '/home/xys/test1/result/chb6/net/0_total/epo64.pt',
              'chb7':'/home/xys/test1/result/chb6/net/0_total/epo26.pt'  ,
              'chb8':'/home/xys/test1/result/chb8/net/0_total/epo66.pt',
              'chb9':'/home/xys/test1/result/chb9/net/0_total/epo46.pt' ,
              'chb10':'/home/xys/test1/result/chb10/net/0_total/epo68.pt',
              # 'chb11':'/home/xys/test1/result/chb11/net/0_total/epo19.pt',
              'chb14':'/home/xys/test1/result/chb14/net/0_total/epo101.pt',
              'chb16':'/home/xys/test1/result/chb16/net/0_total/epo89.pt' ,
              'chb17':'/home/xys/test1/result/chb17/net/0_total/epo38.pt' ,
              'chb18':'/home/xys/test1/result/chb18/net/0_total/epo63.pt' ,
              'chb19':'/home/xys/test1/result/chb19/net/0_total/epo66.pt' ,
              'chb20':'/home/xys/test1/result/chb20/net/0_total/epo53.pt' ,
              'chb21':'/home/xys/test1/result/chb21/net/0_total/epo90.pt' ,
              'chb22':'/home/xys/test1/result/chb22/net/0_total/epo136.pt' ,
              'chb23':'/home/xys/test1/result/chb23/net/0_total/epo30.pt' 

    }




class train_data(Dataset):
    def __init__(self,select,scale,total,TYPE,data_path = 'icachb8'):
        data= []
        label = []
        self.num0 = 0
        self.num1 = 0
        if (TYPE == 'all'):
            ictal_name = '/Z_ictal'
            interitcal_name = '/Z_interIctal'
            preictal_name = '/Z_preIctal'
        elif(TYPE == 'one'):
            ictal_name = '/Z_one_ictal'
            interitcal_name = '/Z_one_interIctal'
            preictal_name = '/Z_one_preIctal'
        elif(TYPE == 'ica'):
            ictal_name = '/ica_ictal'
            interitcal_name = '/ica_interIctal'
            preictal_name = '/ica_preIctal'
        else:
            ictal_name = '/ictal'
            interitcal_name = '/interIctal'
            preictal_name = '/preIctal'
            
            
        interictal = []      
        for i in range(total):
            if i == select :
                continue
            else:
                ictal = np.load('./'+data_path+ ictal_name +str(i)+'.npy')
                preictal = np.load('./'+data_path+ preictal_name +str(i)+'.npy')
                if(len(ictal.shape))==2:
                    ictal = np.expand_dims(ictal,axis=0)
                preictal = np.concatenate((preictal,ictal),0)
                temp = []
                indT = np.arange(preictal.shape[0]*2) 
                
                for s in range(scale):
                    tmp = np.concatenate(np.split(preictal, 2, -1), 0)
                    np.random.seed(args.seed)
                    np.random.shuffle(indT)
                    tmp = tmp[indT]
                    tmp = np.concatenate(np.split(tmp, 2, 0), -1)
                    temp.append(tmp)
                temp.append(preictal)
                temp = np.concatenate(temp, 0)
                preictal = temp
                data.append(preictal)
                label.append(np.ones(preictal.shape[0]))
                self.num1 += preictal.shape[0]   
                temp = np.load('./'+data_path+ interitcal_name +str(i)+'.npy').squeeze()
                
                if(np.array(temp).shape[0] != 0):
                    interictal.append(temp)
                
        interictal =  np.concatenate(interictal,0)            
        ind = np.arange(0,interictal.shape[0])
        np.random.seed(args.seed)
        np.random.shuffle(ind)
        
        
        data.append(interictal[ind[:self.num1]])
        label.append(np.zeros(self.num1))
        self.num0 = self.num1
            
                        
        data = np.concatenate(data,0)
        label = np.concatenate(label,0)
        self.x_data = torch.from_numpy(data).unsqueeze(1)
        self.y_data = torch.from_numpy(label).squeeze()               
                
        
            
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]    
    def __len__(self):
        return self.x_data.size()[0]
    
class test_data(Dataset):
    def __init__(self,select,TYPE,data_path = 'icachb8'):
        if (TYPE == 'all'):
            ictal_name = '/Z_ictal'
            interitcal_name = '/Z_interIctal'
            preictal_name = '/Z_preIctal'
        elif(TYPE == 'one'):
            ictal_name = '/Z_one_ictal'
            interitcal_name = '/Z_one_interIctal'
            preictal_name = '/Z_one_preIctal'
        elif(TYPE == 'ica'):
            ictal_name = '/ica_ictal'
            interitcal_name = '/ica_interIctal'
            preictal_name = '/ica_preIctal'
        else:
            ictal_name = '/ictal'
            interitcal_name = '/interIctal'
            preictal_name = '/preIctal'
        ictal = np.load('./'+data_path+ ictal_name +str(select)+'.npy').squeeze()
        
        interitcal = np.load('./'+data_path+ interitcal_name +str(select)+'.npy').squeeze()
        preictal = np.load('./'+data_path+ preictal_name +str(select)+'.npy').squeeze()
        if(len(ictal.shape))==2:
            ictal = np.expand_dims(ictal,axis=0)
        
        
        print("测试数据为第%d次癫痫发作，发作期数据%d组，前期数据%d组，间期数据%d组"%(select+1,ictal.shape[0],preictal.shape[0],interitcal.shape[0]))
        print(ictal.shape)
        print(preictal.shape) 
        preictal = np.concatenate((preictal,ictal),0)
        self.x_data = torch.from_numpy(np.concatenate((interitcal,preictal),0)).unsqueeze(1)
        self.y_data = torch.from_numpy(np.concatenate((np.zeros(interitcal.shape[0]),np.ones(preictal.shape[0])),0)).unsqueeze(1)
        self.num0 = interitcal.shape[0]
        self.num1 = preictal.shape[0]
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]    
    def __len__(self):
        return self.x_data.size()[0]
    
    
    
    
def compute_kernel(x,y):
    bandlist = [0.25, 0.5, 1, 2, 4]
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = [torch.exp(-(tiled_x - tiled_y).pow(2).sum(2)/float(bandwidth)) for bandwidth in bandlist]
    return kernel_input  #[.., .., .., .., ..] every .. x_size*y_size
def MMD_loss(x,y):
    x = x.squeeze()
    x_kernel = sum(compute_kernel(x, x))      
    y_kernel = sum(compute_kernel(y, y))
    xy_kernel = sum(compute_kernel(x, y))
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

    


def setup_seed(seed):
    
    torch.manual_seed(seed)#给CPU设置种子
    torch.cuda.manual_seed(seed)#给当前GPU设置种子
    torch.cuda.manual_seed_all(seed)#
    torch.backends.cudnn.deterministic = True    
    
    
parser = argparse.ArgumentParser(description='命令行中传入作为测试集的发作次数')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--count', type=int ,help='癫痫次数',default = 99)
parser.add_argument('--ic', type=int ,help='ic数',default = 22)
parser.add_argument('patient', type=str ,help='选择的病人')
parser.add_argument('--type', type=str ,help='预处理类型',default = 'no' )
parser.add_argument('--gpu', type=str ,help='GPU号',default = '0' )
parser.add_argument('--seed', type=int ,help='随机种子',default = 5 )
parser.add_argument('--scale', type=int ,help='上采样倍率',default = 1 )
parser.add_argument('--batch', type=int ,help='batch_size',default = 512 )
args = parser.parse_args()    


batch_size = args.batch
decay = 0.9
weight_decay =  5e-4
epochs = 500
loop = 8



if __name__ == "__main__":
    if(torch.cuda.is_available()):###如果有GPU可以用
        device = torch.device("cuda:"+args.gpu)
        print("Train Mode : One GPU;", device)
        for chb in data_check[args.patient]:
            if (args.count != 99):
                chb = args.count
            text_path = './result/'+args.patient+'/result/'+str(chb)+'_TL/pool.txt'
            train = train_data(chb,args.scale,data_total[args.patient],args.type,'ica'+args.patient)
            test = test_data(chb,args.type,'ica'+args.patient)
            num0 = train.num0
            num1 = train.num1
            
            indices = list(range(train.__len__()))
            data_len = train.__len__()
            np.random.seed(args.seed)
            np.random.shuffle(indices)
            val_indices = indices[int(data_len*0.8):]
            train_indices = indices[:int(data_len*0.8)]
            train_len = int(data_len*0.8)
            val_len = data_len-train_len
            
            test_indices = np.array(list(range(test.__len__())))
            train_indices,val_indices = np.array(train_indices),np.array(val_indices)
        
            Train = Data.Subset(train, train_indices)
            Val = Data.Subset(train, val_indices)
            Test = Data.Subset(test, test_indices)
            
            train_loader = Data.DataLoader(Train,
                                            batch_size = batch_size,
                                            shuffle = False)
            val_loader = Data.DataLoader(Val,
                                          batch_size = batch_size,
                                          shuffle = False)
            test_loader = Data.DataLoader(Test,
                                          batch_size = batch_size,
                                          shuffle = False)
            
            
            
            net_agent = SeizureNet(device,64,args.ic,128).to(device)
            
            
            path = chb_path[args.patient]
            input_dict =torch.load(path)#导入迁移的参数
            net_agent.load_state_dict(input_dict)
            
            setup_seed(args.seed)
            net = SeizureNet(device,64,args.ic,128).to(device)
            raw_dict = net.state_dict()#原始的参数
            
            #开始迁移
            model_dict = net_agent.featureExtractor.state_dict()
            input_state_dict = {'featureExtractor.'+k:v for k,v in model_dict.items()}
            raw_dict.update(input_state_dict)#更新
            for l in range(loop+1):#比较不同冻结层数的差别
                if l >= 1:    
                    net.load_state_dict(raw_dict)
                if l >=1:
                    for name,para in net.featureExtractor.embedding.named_parameters():
                        if name in ['0.conv_input.weight',
                                    '0.bn_input.weight',
                                    '0.bn_input.bias']:
    
                            para.requires_grad = False
                if l>=2:
                    for name,para in net.featureExtractor.embedding.named_parameters():
                        if name in ['1.conv_expand.weight',
                                    '1.conv1.weight',
                                    '1.bn1.weight',
                                    '1.bn1.bias']:
                            
    
                            para.requires_grad = False
                
                if l>=3:
                    for name,para in net.featureExtractor.embedding.named_parameters():
                        if name in ['1.conv2.weight',
                                    '1.bn2.weight',
                                    '1.bn2.bias']:
                            
                            para.requires_grad = False
                if l>=4:
                    for name,para in net.featureExtractor.downsampled_gamma.named_parameters():
                            para.requires_grad = False
                    for name,para in net.featureExtractor.downsampled_beta.named_parameters():
                            para.requires_grad = False
                    for name,para in net.featureExtractor.downsampled_alpha.named_parameters():
                            para.requires_grad = False
                    for name,para in net.featureExtractor.downsampled_theta.named_parameters():
                            para.requires_grad = False
                    for name,para in net.featureExtractor.downsampled_delta.named_parameters():
                            para.requires_grad = False
                            
                if l>=5:
                    for name,para in net.featureExtractor.SEGamma.named_parameters():
                        if name in ['conv0.weight',
                                    'bn0.weight',
                                    'bn0.bias',
                                    'se0.fc.0.weight',
                                    'se0.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEBeta.named_parameters():
                        if name in ['conv0.weight',
                                    'bn0.weight',
                                    'bn0.bias',
                                    'se0.fc.0.weight',
                                    'se0.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEAlpha.named_parameters():
                        if name in ['conv0.weight',
                                    'bn0.weight',
                                    'bn0.bias',
                                    'se0.fc.0.weight',
                                    'se0.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEDelta.named_parameters():
                        if name in ['conv0.weight',
                                    'bn0.weight',
                                    'bn0.bias',
                                    'se0.fc.0.weight',
                                    'se0.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SETheta.named_parameters():
                        if name in ['conv0.weight',
                                    'bn0.weight',
                                    'bn0.bias',
                                    'se0.fc.0.weight',
                                    'se0.fc.2.weight']:
                            
                            para.requires_grad = False
                if l>=6:
                    for name,para in net.featureExtractor.SEGamma.named_parameters():
                        if name in ['conv1.weight',
                                    'bn1.weight',
                                    'bn1.bias',
                                    'se1.fc.0.weight',
                                    'se1.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEBeta.named_parameters():
                        if name in ['conv1.weight',
                                    'bn1.weight',
                                    'bn1.bias',
                                    'se1.fc.0.weight',
                                    'se1.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEAlpha.named_parameters():
                        if name in ['conv1.weight',
                                    'bn1.weight',
                                    'bn1.bias',
                                    'se1.fc.0.weight',
                                    'se1.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEDelta.named_parameters():
                        if name in ['conv1.weight',
                                    'bn1.weight',
                                    'bn1.bias',
                                    'se1.fc.0.weight',
                                    'se1.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SETheta.named_parameters():
                        if name in ['conv1.weight',
                                    'bn1.weight',
                                    'bn1.bias',
                                    'se1.fc.0.weight',
                                    'se1.fc.2.weight']:
                            
                            para.requires_grad = False
                if l>=7:
                    for name,para in net.featureExtractor.SEGamma.named_parameters():
                        if name in ['conv2.weight',
                                    'bn2.weight',
                                    'bn2.bias',
                                    'se2.fc.0.weight',
                                    'se2.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEBeta.named_parameters():
                        if name in ['conv2.weight',
                                    'bn2.weight',
                                    'bn2.bias',
                                    'se2.fc.0.weight',
                                    'se2.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEAlpha.named_parameters():
                        if name in ['conv2.weight',
                                    'bn2.weight',
                                    'bn2.bias',
                                    'se2.fc.0.weight',
                                    'se2.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SEDelta.named_parameters():
                        if name in ['conv2.weight',
                                    'bn2.weight',
                                    'bn2.bias',
                                    'se2.fc.0.weight',
                                    'se2.fc.2.weight']:
                            
                            para.requires_grad = False
                    for name,para in net.featureExtractor.SETheta.named_parameters():
                        if name in ['conv2.weight',
                                    'bn2.weight',
                                    'bn2.bias',
                                    'se2.fc.0.weight',
                                    'se2.fc.2.weight']:
                            
                            para.requires_grad = False
                if l >=8:
                    
                    for name,para in net.featureExtractor.SEA.named_parameters():
                        para.requires_grad = False
                    for name,para in net.featureExtractor.SEB.named_parameters():
                        para.requires_grad = False
                    for name,para in net.featureExtractor.SED.named_parameters():
                        para.requires_grad = False
                    for name,para in net.featureExtractor.SET.named_parameters():
                        para.requires_grad = False
                    for name,para in net.featureExtractor.SEG.named_parameters():
                        para.requires_grad = False
            
    
                
                learning_rate = 0.001
                f = open(text_path,'a')  
                for name,para in net.named_parameters():
                    if para.requires_grad == False:
                        print('已冻结层数：'+name)
                        f.write('已冻结层数：'+name+'\n')
                
                
                net.load_state_dict(raw_dict)
                #只导入未冻结的参数进优化器
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,weight_decay=weight_decay)
                
                # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
                #定义损失函数针对不平衡样本，对数目更少的类别给出更大的误差惩罚权重
                
                weight_CE=torch.FloatTensor([num0,num1])
                weight_CE=weight_CE.to(device)
                criteon = nn.CrossEntropyLoss(weight=weight_CE) 
                Loss =[]
                confusion_matrix = []
                test_loss = []
                test_confusion = []
                softmax = torch.nn.Softmax(dim = 1)
                fpr = []
                tpr = []
                
                
                min_val_loss = 100000
                max_val_acc =0
                end_count =0
                save_epo = 0
                
                for epo in range(epochs):#开始训练
                    net.train()
                    if(epo+1)%10 == 0:
                        learning_rate = learning_rate*decay
                    optimizer.zero_grad()
                    train_loss = 0
                    train_correct = 0
                    #############################################
                    #                  混淆矩阵
                    #           真实为真     真实为假       
                    #预测为真       TP          FP
                    #
                    #预测为假       FN          TF
                    #          Recall=TP/(TP+FN)
                    #          Precision = TP/(TP+FP)
                    #          Accuracy = (TP+TN)/(TP+TN+FP+FN)
                    #          F1Sorce = 2*Precision*Recall/(Precision+Recall)
                    #############################################
                    TP = 0
                    FN = 0
                    FP = 0
                    TN = 0
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = Variable(data).to(device), Variable(target).to(device)
                        data = data.type(torch.cuda.FloatTensor)
                        output,source_feature,source_y= net(data)
                        _, predicted = torch.max(output, 1)
                        train_correct += (predicted == target.squeeze()).sum().item()
                        TP_temp = ((predicted == target.squeeze())*target.squeeze()).sum().item()
                        FP_temp = target.squeeze().sum().item() - TP_temp
                        TN_temp = (predicted == target.squeeze()).sum().item() - TP_temp
                        FN_temp = len(target.squeeze()) - TP_temp-TN_temp-FP_temp
                        FN += FN_temp
                        FP += FP_temp 
                        TP += TP_temp
                        TN += TN_temp
                        loss =criteon(output, target.long())
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
        
                    f = open(text_path,'a')    
                    print("Epoch%03d: Training_loss = %.5f" % (epo + 1, train_loss))  
                    f.write("Epoch%03d: Training_loss = %.5f\n" % (epo + 1, train_loss))                
                    # print("Epoch%03d: TestSet_Accuracy = %.3f" % (epoch + 1, float(val_correct /(l-sum(mat_data.info[-1])))))
                    print(" Epoch%03d: RrainSet_Accuracy = %.3f" % (epo + 1, float(train_correct /train_len)))
                    f.write(" Epoch%03d: RrainSet_Accuracy = %.3f\n" % (epo + 1, float(train_correct /train_len)))
                    print(np.array([[TP,FP],
                                    [FN,TN]]))
                    f.write(str([TP,FP])+'\n')
                    f.write(str([FN,TN])+'\n')
                    f.close()
                    Loss.append([epo + 1, train_loss])
                    confusion_matrix.append([
                                              epo+1,
                                              np.array([[TP,FP],
                                                        [FN,TN]])
                                              ])
                    if(epo+1)%1 == 0:
                        net.eval()
                        #使用验证集 验证准确率 防止过拟合
                        val_loss=0
                        val_correct=0
                        sensitivity = 0
                        TP = 0
                        FN = 0
                        FP = 0
                        TN = 0
                        score = []
                        y = []
                        with torch.no_grad():
                            for batch_idx, (data, target) in enumerate(val_loader):
                                data, target = Variable(data).to(device), Variable(target).to(device)
                                data = data.type(torch.cuda.FloatTensor)
                            
                                output,_,_= net(data)
                                _, predicted = torch.max(output, 1)
                                val_correct += (predicted == target.squeeze()).sum().item()
                                TP_temp = ((predicted == target.squeeze())*target.squeeze()).sum().item()
                                FP_temp = target.squeeze().sum().item() - TP_temp
                                TN_temp = (predicted == target.squeeze()).sum().item() - TP_temp
                                FN_temp = len(target.squeeze()) - TP_temp-TN_temp-FP_temp
                                FN += FN_temp
                                FP += FP_temp 
                                TP += TP_temp
                                TN += TN_temp
                                loss =criteon(output, target.long())
                                val_loss += loss.item()
                                out = softmax(output)
                                score += list(out[:,1].cpu().detach().numpy() )
                                y += list(target.cpu().detach().numpy())
        
        
                        f = open(text_path,'a')    
                        print("###" * 15)
                        # print("Epoch%03d: TestSet_Accuracy = %.3f" % (epoch + 1, float(val_correct /(train_num_1-train_num_2))))
                        f.write("Epoch%03d: TestSet_Accuracy = %.3f\n" % (epo + 1, float(val_correct /val_len)))
                        print("Epoch%03d: TestSet_Accuracy = %.3f" % (epo + 1, float(val_correct /val_len)))
                        f.write("Epoch%03d: TestSet_loss = %.3f\n" % (epo + 1, float(val_loss)))
                        print("Epoch%03d: TestSet_loss = %.3f" % (epo + 1, float(val_loss)))
                        print([TP,FP])
                        f.write(str([TP,FP])+'\n')
                        f.write(str([FN,TN])+'\n')
                        print([FN,TN])
                        print("###" * 15)
                        f.close()
                        if(min_val_loss>val_loss):
                            min_val_loss = val_loss
                            end_count = 0 
                            save_epo = epo+1
                            save_state = net.state_dict()
                        else:
                            end_count += 1 
                            
                        if end_count >24:
                            f = open(text_path,'a')  
                            print('train finish')
                            print('the epo %d is the best'%(save_epo+1))
                            f.write('train finish\nthe epo %d is the best\n'%(save_epo+1))
                            f.close()
                            np.save('./result/'+args.patient+'/result/'+str(chb)+'_TL/'+str(l)+'_train_loss.npy',Loss)
                            np.save('./result/'+args.patient+'/result/'+str(chb)+'_TL/'+str(l)+'_confusion_train.npy',confusion_matrix)
                            np.save('./result/'+args.patient+'/result/'+str(chb)+'_TL/'+str(l)+'_confusion_val.npy',test_confusion)
                            np.save('./result/'+args.patient+'/result/'+str(chb)+'_TL/'+str(l)+'_val_loss.npy',test_loss)
                            path = './result/'+args.patient+'/net/'+str(chb)+'_TL/l'+str(l)+'_best.pt'
    
                            torch.save(save_state, path)
                            break
                            
                        test_loss.append([epo + 1, val_loss])
                        test_confusion.append([
                                              epo+1,
                                              np.array([[TP,FP],
                                                        [FN,TN]])
                                              ])
                        
                        
                net.load_state_dict(save_state)       
                
                flag_label_0 = 0
                flag_label_1 = 0 
                with torch.no_grad():
                    # for batch_idx, (data, target) in enumerate(val_loader):
                    #             data, target = Variable(data).to(device), Variable(target).to(device)
                    #             data = data.type(torch.cuda.FloatTensor)
                            
                    #             output,feature,feature_y= net(data)
                    #             for label,f, y in zip(target,feature,feature_y):
                    #                 if(label == 1):
                    #                     if(flag_label_1 == 0):
                    #                         f_1 = f.detach().unsqueeze(0)
                    #                         y_1 = y.detach().unsqueeze(0)
                    #                         flag_label_1 =1
                    #                     else:
                    #                         f_1 = torch.cat((f_1,f.detach().unsqueeze(0)),0)
                    #                         y_1 = torch.cat((y_1,y.detach().unsqueeze(0)),0)
                                            
                    #                 else:
                    #                     if(flag_label_0 == 0):
                    #                         f_0 = f.detach().unsqueeze(0)
                    #                         y_0 = y.detach().unsqueeze(0)
                    #                         flag_label_0 =1
                    #                     else:
                    #                         f_0 = torch.cat((f_0,f.detach().unsqueeze(0)),0)
                    #                         y_0 = torch.cat((y_0,y.detach().unsqueeze(0)),0)
                                            
                                    
                    # mmd_f = MMD_loss(f_0, f_1)   
                    # mmd_y = MMD_loss(y_0, y_1)
                    # f = open(text_path,'a') 
                    # f.write('feature的label间MMD距离为：%.7f,y的label间MMD距离为：%.7f'%(mmd_f,mmd_y))
                    # f.close()
                    
                    # print('feature的label间MMD距离为：%.7f,y的label间MMD距离为：%.7f'%(mmd_f,mmd_y))
                    test_correct = 0
                    sensitivity = 0
                    TP = 0
                    FN = 0
                    FP = 0
                    TN = 0
                    score = []
                    y = []
                    y_pre = []
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = Variable(data).to(device), Variable(target).to(device)
                        data = data.type(torch.cuda.FloatTensor).squeeze().unsqueeze(1)
                        output,_,_= net(data)
                        _, predicted = torch.max(output, 1)
                        test_correct += (predicted == target.squeeze()).sum().item()
                        TP_temp = ((predicted == target.squeeze())*target.squeeze()).sum().item()
                        FP_temp = target.squeeze().sum().item() - TP_temp
                        TN_temp = (predicted == target.squeeze()).sum().item() - TP_temp
                        FN_temp = len(target.squeeze()) - TP_temp-TN_temp-FP_temp
                        FN += FN_temp
                        FP += FP_temp 
                        TP += TP_temp
                        TN += TN_temp
                        out = softmax(output)
                        score += list(out[:,1].cpu().detach().numpy() )
                        y += list(target.cpu().detach().numpy())
                        y_pre += list(predicted.cpu().detach().numpy())
        
                    y,score,y_pre = np.array(y),np.array(score),np.array(y_pre)         
                    FPR, TPR, thresholds = roc_curve(y, score)
                    AUC = auc(FPR,TPR)
                    f = open(text_path,'a') 
                    f.write("TestSet_Accuracy = %.3f\n" % ( float(test_correct /test.__len__())))
                    print("TestSet_Accuracy = %.3f" % ( float(test_correct /test.__len__())))
                    print("TestSet_AUC = %.3f" % ( AUC))
                    f.write("TestSet_AUC = %.3f\n" % ( AUC))
                    print(np.array([[TP,FP],
                                    [FN,TN]]))    
                    f.write(str([TP,FP])+'\n')
                    f.write(str([FN,TN])+'\n')
                    f.close()
                    np.save('./result/'+args.patient+'/result/'+str(chb)+'_TL/l'+str(l)+'_y_pre.npy',[y_pre.reshape(1,-1),score.reshape(1,-1),y.reshape(1,-1)])
            if (args.count != 99):
                break
            