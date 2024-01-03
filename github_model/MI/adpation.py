from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import Dataset
import torch
import numpy as np
import GMM

import scipy.stats as ss

def JS_divergence(p, q, base):
    index =  list(range(p.shape[0]))
    np.random.seed(123252)
    np.random.shuffle(index)
    p = p[index[:q.shape[0]]]
    M = (p+q)/2
    return 0.5 * ss.entropy(p, M, base=base) + 0.5 * ss.entropy(q, M, base=base)



batch = 128

class torch_feature_data(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.x_data.size()[0]

class adpation():

    def __init__(self,eeg_type,subject_id,model,train_set,source_set,device) -> None:
        self.eeg_type = eeg_type
        self.subject_id = subject_id
        self.model = model
        self.data = train_set.X
        self.label = train_set.y
        self.source_data = source_set.X
        self.source_label = source_set.y
        self.l = -1
        self.device = device
        model.feature_flag = True
        if(self.eeg_type == "MI"):
            self.l = self.MI()
        if(self.eeg_type == "Seizure"):
            self.l = self.Seizure()
    def MI(self):
        trian_data = torch_feature_data(self.data,self.label)
        indices = list(range(trian_data.__len__()))
        Train = Data.Subset(trian_data, indices)
        train_loader = Data.DataLoader(Train,
                                        batch_size = batch,
                                        shuffle = False)
        source_data = torch_feature_data(self.source_data,self.source_label)
        indices = list(range(source_data.__len__()))
        Source = Data.Subset(source_data, indices)
        source_loader = Data.DataLoader(Source,
                                        batch_size = batch,
                                        shuffle = False)
        self.model.eval()
        shallow = []
        deep = []
        s_shallow =[]
        s_deep = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                data = data.type(torch.cuda.FloatTensor)
                s,d= self.model(data)
                shallow.append(s.cpu().detach().numpy())
                deep.append(d.cpu().detach().numpy())
            shallow = np.concatenate(shallow)
            deep = np.concatenate(deep)
        
            for batch_idx, (data, target) in enumerate(source_loader):
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                data = data.type(torch.cuda.FloatTensor)
                s,d= self.model(data) 
                s_shallow.append(s.cpu().detach().numpy())
                s_deep.append(d.cpu().detach().numpy())
            s_shallow = np.concatenate(s_shallow)
            s_deep = np.concatenate(s_deep)   
        transfer_shallow = self.cal_transfer_score(shallow,s_shallow)
        transfer_deep = self.cal_transfer_score(deep,s_deep)

        if(transfer_shallow[0]/transfer_deep[0]<0.8):
            self.l = 0 
        elif (transfer_shallow[0]/transfer_deep[0]>1.25):
            self.l = 1
        elif (transfer_shallow[1]/transfer_deep[1]<0.8):
            self.l = 0
        elif (transfer_shallow[1]/transfer_deep[1]>1.25):
            self.l = 1
        else:
            self.l = 0
        

    def Seizure(self):
        trian_data = torch_feature_data(self.data,self.label)
        indices = list(range(trian_data.__len__()))
        Train = Data.Subset(trian_data, indices)
        train_loader = Data.DataLoader(Train,
                                        batch_size = 128,
                                        shuffle = False)
        source_data = torch_feature_data(self.source_data,self.source_label)
        indices = list(range(source_data.__len__()))
        Source = Data.Subset(source_data, indices)
        source_loader = Data.DataLoader(Source,
                                        batch_size = batch,
                                        shuffle = False)
        self.model.eval()
        gamma = []
        beta = []
        alpha = []
        theta = []
        delta = []
        s_gamma = []
        s_beta = []
        s_alpha = []
        s_theta = []
        s_delta = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                data = data.type(torch.cuda.FloatTensor)
                g,b,a,t,d= self.model(data)
                gamma.append(g.cpu().detach().numpy())
                beta.append(b.cpu().detach().numpy())
                alpha.append(a.cpu().detach().numpy())
                theta.append(t.cpu().detach().numpy())
                delta.append(d.cpu().detach().numpy())
            gamma = np.concatenate(gamma)
            beta = np.concatenate(beta)
            alpha = np.concatenate(alpha)
            theta = np.concatenate(theta)
            delta = np.concatenate(delta)

            for batch_idx, (data, target) in enumerate(source_loader):
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                data = data.type(torch.cuda.FloatTensor)
                g,b,a,t,d= self.model(data)
                s_gamma.append(g.cpu().detach().numpy())
                s_beta.append(b.cpu().detach().numpy())
                s_alpha.append(a.cpu().detach().numpy())
                s_theta.append(t.cpu().detach().numpy())
                s_delta.append(d.cpu().detach().numpy())
            s_gamma = np.concatenate(s_gamma)
            s_beta = np.concatenate(s_beta)
            s_alpha = np.concatenate(s_alpha)
            s_theta = np.concatenate(s_theta)
            s_delta = np.concatenate(s_delta)
        transfer_gamma = self.cal_transfer_score(gamma,s_gamma)
        transfer_beta =  self.cal_transfer_score(beta,s_beta)
        transfer_alpha =  self.cal_transfer_score(alpha,s_alpha)
        transfer_theta =  self.cal_transfer_score(theta ,s_theta)
        transfer_delta = self.cal_transfer_score(delta,s_delta)
        gmm = GMM([transfer_gamma,transfer_beta,transfer_alpha,transfer_theta,transfer_delta],[],2)

        for i in gmm.assignments:
            self.l +=1
            if(i == 0):
                break
    def classDistance(self,label,feature):
        idx = np.where(self.label == label)[0]
        other_idx = np.where(self.label != label)[0]
        center = np.mean(feature[idx])
        center_2 = np.mean(feature[other_idx])
        inner  = np.linalg.norm(feature[idx] -center,axis=0).sum()
        intra = np.linalg.norm(feature[other_idx] -center,axis=0).sum()
        intra += np.linalg.norm(feature[idx] -center_2,axis=0).sum()
        return inner/intra
        

        
    def cal_transfer_score(self,feature,source_feature):
        S2 = JS_divergence(feature,source_feature,2)
        if(self.eeg_type =="MI"):
            S1 = self.classDistance(0,feature)
            S1 += self.classDistance(1,feature)
        elif(self.eeg_type == "Seizure"):
            S1 = self.classDistance(0,feature)
            S1 += self.classDistance(1,feature)
            S1 += self.classDistance(2,feature)
            S1 += self.classDistance(3,feature) 
        else:
            S1 = 0
        return [S1,S2]

