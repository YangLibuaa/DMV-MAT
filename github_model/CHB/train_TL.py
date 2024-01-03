# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:50:58 2020

@author: XYS
"""
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append('D:\\goodgoodstudy\\useful') 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import getdata
import argparse
from model import SeizureNet
import EEGNet


lenth = 5
batch_size = 512
learning_rate = 0.001
decay = 0.9
weight_decay =  5e-4
epochs = 500
M = 60
LAMBDA = 1

data_total = {'chb1':7,
              'chb2':3,
              'chb3':7,
              'chb4':4,
              'chb5':5,
              'chb6':10,
              'chb7':3,
              'chb8':5,
              'chb9':4,
              'chb10':7
               } 
    
    
class train_data(Dataset):
    def __init__(self,select,scale,total,TYPE,data_path = 'icachb8'):
        data= []
        label = []
        self.num0 = 0
        self.num1 = 0
        for i in range(total):
            if i == select :
                continue
            else:
                if (TYPE == 'all'):
                    ictal_name = '/Z_ictal'
                    interitcal_name = '/Z_interIctal'
                    preictal_name = '/Z_preIctal'
                elif(TYPE == 'one'):
                    ictal_name = '/Z_one_ictal'
                    interitcal_name = '/Z_one_interIctal'
                    preictal_name = '/Z_one_preIctal'
                else:
                    ictal_name = '/ictal'
                    interitcal_name = '/interIctal'
                    preictal_name = '/preIctal'
                ictal = np.load('./'+data_path+ ictal_name +str(i)+'.npy')
                interitcal = np.load('./'+data_path+ interitcal_name +str(i)+'.npy')
                preictal = np.load('./'+data_path+ preictal_name +str(i)+'.npy')
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
                
                ind = np.arange(0,interitcal.shape[0])
                
                np.random.seed(args.seed)
                np.random.shuffle(ind)
                data.append(interitcal[ind[:preictal.shape[0]]])
                label.append(np.zeros(preictal.shape[0]))
                self.num0 += preictal.shape[0]
                
                data.append(preictal)
                label.append(np.ones(preictal.shape[0]))
                self.num1 += preictal.shape[0]
                
                
        data = np.concatenate(data,0)
        label = np.concatenate(label,0)
        self.x_data = torch.from_numpy(data).unsqueeze(1)
        self.y_data = torch.from_numpy(label).squeeze()               
                
        
            
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]    
    def __len__(self):
        return self.x_data.size()[0]
    

    
def setup_seed(seed):
    
    torch.manual_seed(seed)#给CPU设置种子
    torch.cuda.manual_seed(seed)#给当前GPU设置种子
    torch.cuda.manual_seed_all(seed)#
    torch.backends.cudnn.deterministic = True
    
    
def judge(source,target):
    '''
    

    Parameters
    ----------
    source : 源字符串
        DESCRIPTION.
    target : 检查源字符串里面存在的目标
        DESCRIPTION.

    Returns
    -------
    None.

    '''    
    flag = True
    for i in range(len(target)):
        if(source[i] != target[i]):
            flag = False
            break
    return flag
        


parser = argparse.ArgumentParser(description='命令行中传入作为测试集的发作次数')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('count', type=int ,help='癫痫次数')
parser.add_argument('--ic', type=int ,help='ic数',default = 22)
parser.add_argument('patient', type=str ,help='选择的病人')
parser.add_argument('--type', type=str ,help='预处理类型',default = 'no' )
parser.add_argument('--gpu', type=str ,help='GPU号',default = '0' )
parser.add_argument('--seed', type=int ,help='随机种子',default = 5 )
parser.add_argument('--scale', type=int ,help='上采样倍率',default = 1 )
args = parser.parse_args()    

if __name__ == "__main__":
    root,files = getdata.file_name('data')
    seizure_num = len(files)
    if(torch.cuda.is_available()):###如果有GPU可以用
        device = torch.device("cuda:"+args.gpu)
        print("Train Mode : One GPU;", device)
        
        train_data = train_data(args.count,args.scale,data_total[args.patient],args.type,'ica'+args.patient)
        indices = list(range(train_data.__len__()))
        data_len = train_data.__len__()
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        test_indices = indices[int(data_len*0.8):]
        train_indices = indices[:int(data_len*0.8)]
        train_len = int(data_len*0.8)
        test_len = data_len-train_len
        
        train_indices,test_indices = np.array(train_indices),np.array(test_indices)
    
        train = Data.Subset(train_data, train_indices)
        test = Data.Subset(train_data, test_indices)
        train_loader = Data.DataLoader(train,
                                        batch_size = batch_size,
                                        shuffle = False)
        test_loader = Data.DataLoader(test,
                                      batch_size = batch_size,
                                      shuffle = False)
        
        
        setup_seed(args.seed)
        # net = setmodule.stSE_HGCN(22).to(device)
        net = SeizureNet(device,64,args.ic,128).to(device)
        net_agent = SeizureNet(device,64,args.ic,128).to(device)
        # net = EEGNet.EEGNet(channels=args.ic,samples=1280,n_classes = 2).to(device)
        
        raw_dict = net.state_dict()#原始的参数
        if(args.patient=='chb1'):
            path = '/home/xys/test1/' + 'result/chb1/net/0_total/count7epo31.pt'
        elif(args.patient=='chb2'):
            path = '/home/xys/test1/' + 'result/chb2/net/0_total/count5epo158.pt'
        elif(args.patient=='chb3'):
            path = '/home/xys/test1/' + 'result/chb3/net/0_total/count7epo72.pt'
        elif(args.patient=='chb4'):
            path = '/home/xys/test1/' + 'result/chb4/net/0_total/count2epo50.pt'
        elif(args.patient=='chb5'):    
            path = '/home/xys/test1/' + 'result/chb5/net/0_total/count4epo137.pt'
        input_dict =torch.load(path)#导入迁移的参数
        net_agent.load_state_dict(input_dict)
        
        
        #导入并冻结
        model_dict = net_agent.featureExtractor.embedding.state_dict()
        input_state_dict = {'featureExtractor.embedding.'+k:v for k,v in model_dict.items()}
        for para in net.featureExtractor.embedding.parameters():
            para.requires_grad = False 
        raw_dict.update(input_state_dict)#更新
        
        
        model_dict = net_agent.featureExtractor.downsampled_gamma.state_dict()
        input_state_dict = {'featureExtractor.downsampled_gamma.'+k:v for k,v in model_dict.items()}
        
        raw_dict.update(input_state_dict)#更新
        
        model_dict = net_agent.featureExtractor.downsampled_beta.state_dict()
        input_state_dict = {'featureExtractor.downsampled_beta.'+k:v for k,v in model_dict.items()}
        
        raw_dict.update(input_state_dict)#更新
        
        model_dict = net_agent.featureExtractor.downsampled_alpha.state_dict()
        input_state_dict = {'featureExtractor.downsampled_alpha.'+k:v for k,v in model_dict.items()}
        
        raw_dict.update(input_state_dict)#更新
        
        model_dict = net_agent.featureExtractor.downsampled_theta.state_dict()
        input_state_dict = {'featureExtractor.downsampled_theta.'+k:v for k,v in model_dict.items()}
        
        raw_dict.update(input_state_dict)#更新
        
        model_dict = net_agent.featureExtractor.downsampled_delta.state_dict()
        input_state_dict = {'featureExtractor.downsampled_delta.'+k:v for k,v in model_dict.items()}
        
        raw_dict.update(input_state_dict)#更新
        
        # for para in net.featureExtractor.downsampled_delta.parameters():
        #     para.requires_grad = False 
        # for para in net.featureExtractor.downsampled_theta.parameters():
        #     para.requires_grad = False 
        # for para in net.featureExtractor.downsampled_alpha.parameters():
        #     para.requires_grad = False 
        # for para in net.featureExtractor.downsampled_beta.parameters():
        #     para.requires_grad = False 
        # for para in net.featureExtractor.downsampled_gamma.parameters():
        #     para.requires_grad = False 
        # # 导入不冻结
        # model_dict = net_agent.featureExtractor.SEGamma.state_dict()
        # input_state_dict = {'featureExtractor.SEGamma.'+k:v for k,v in model_dict.items()}
        # raw_dict.update(input_state_dict)#更新
        
        # model_dict = net_agent.featureExtractor.SEBeta.state_dict()
        # input_state_dict = {'featureExtractor.SEBeta.'+k:v for k,v in model_dict.items()}
        # raw_dict.update(input_state_dict)#更新
        
        # model_dict = net_agent.featureExtractor.SEAlpha.state_dict()
        # input_state_dict = {'featureExtractor.SEAlpha.'+k:v for k,v in model_dict.items()}
        # raw_dict.update(input_state_dict)#更新
        
        # model_dict = net_agent.featureExtractor.SEDelta.state_dict()
        # input_state_dict = {'featureExtractor.SEDelta.'+k:v for k,v in model_dict.items()}
        # raw_dict.update(input_state_dict)#更新
        
        # model_dict = net_agent.featureExtractor.SETheta.state_dict()
        # input_state_dict = {'featureExtractor.SETheta.'+k:v for k,v in model_dict.items()}
        # raw_dict.update(input_state_dict)#更新
        
        net.load_state_dict(raw_dict)
        #只导入未冻结的参数进优化器
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,weight_decay=weight_decay)
        
        # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #定义损失函数针对不平衡样本，对数目更少的类别给出更大的误差惩罚权重
        num0 = train_data.num0
        num1 = train_data.num1
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
                output,_,_ = net(data)
                # output = net(data)
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

                
            print("Epoch%03d: Training_loss = %.5f" % (epo + 1, train_loss))  
            # print("Epoch%03d: TestSet_Accuracy = %.3f" % (epoch + 1, float(val_correct /(l-sum(mat_data.info[-1])))))
            print("Epoch%03d: RrainSet_Accuracy = %.3f" % (epo + 1, float(train_correct /train_len)))
            print(np.array([[TP,FP],
                            [FN,TN]]))
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
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = Variable(data).to(device), Variable(target).to(device)
                        data = data.type(torch.cuda.FloatTensor)
                        output,_,_ = net(data)
                        # output = net(data)
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


                print("###" * 15)
                # print("Epoch%03d: TestSet_Accuracy = %.3f" % (epoch + 1, float(val_correct /(train_num_1-train_num_2))))
                print("Epoch%03d: TestSet_Accuracy = %.3f" % (epo + 1, float(val_correct /test_len)))
                print([TP,FP])
                print([FN,TN])
                print("###" * 15)
                
                if(min_val_loss>val_loss):
                    min_val_loss = val_loss
                    end_count = 0 
                    save_epo = epo+1
                else:
                    end_count += 1 
                    
                if end_count >14:
                    if(save_epo>10):
                        print('train finish')
                        print('the epo %d is the best'%(save_epo+1))
                        break
                test_loss.append([epo + 1, val_loss])
                test_confusion.append([
                                      epo+1,
                                      np.array([[TP,FP],
                                                [FN,TN]])
                                      ])
                
                np.save('./result/'+args.patient+'/result/'+str(args.count)+'_TL/train_loss.npy',Loss)
                np.save('./result/'+args.patient+'/result/'+str(args.count)+'_TL/confusion_train.npy',confusion_matrix)
                np.save('./result/'+args.patient+'/result/'+str(args.count)+'_TL/confusion_val.npy',test_confusion)
                np.save('./result/'+args.patient+'/result/'+str(args.count)+'_TL/test_loss.npy',test_loss)
                path = './result/'+args.patient+'/net/'+str(args.count)+'_TL/epo'+str(epo+1)+'.pt'
                
                
                # np.save('./result/'+args.patient+'/result/'+str(args.count)+'_TL/train_loss_2.npy',Loss)
                # np.save('./result/'+args.patient+'/result/'+str(args.count)+'_TL/confusion_train_2.npy',confusion_matrix)
                # np.save('./result/'+args.patient+'/result/'+str(args.count)+'_TL/confusion_val_2.npy',test_confusion)
                # np.save('./result/'+args.patient+'/result/'+str(args.count)+'_TL/test_loss_2.npy',test_loss)
                # path = './result/'+args.patient+'/net/'+str(args.count)+'_TL/count'+str(count+1)+'epo'+str(epo+1)+'_2.pt'
                
                
                # np.save('./eegnet/'+args.patient+'/result/'+str(args.count)+'_TL/train_loss.npy',Loss)
                # np.save('./eegnet/'+args.patient+'/result/'+str(args.count)+'_TL/confusion_train.npy',confusion_matrix)
                # np.save('./eegnet/'+args.patient+'/result/'+str(args.count)+'_TL/confusion_val.npy',test_confusion)
                # np.save('./eegnet/'+args.patient+'/result/'+str(args.count)+'_TL/test_loss.npy',test_loss)
                # path = './eegnet/'+args.patient+'/net/'+str(args.count)+'_TL/count'+str(count+1)+'epo'+str(epo+1)+'.pt'
                
                
                
                torch.save(net.state_dict(), path)
                    
                    
            