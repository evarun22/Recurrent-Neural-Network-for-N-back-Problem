import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime
np.random.seed(34)
class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        
        

    def backward (self, yhat,y,zt,ht,xs):
        
        lr=0.00061
        self.w=self.w.reshape((numHidden,numInput))
        dU=np.zeros((self.numHidden,self.numHidden))
        for t in range(len(zt)-1,0,-1):
            #print(t)
            gt_trans=(1-(np.tanh(zt[t]))**2).T
            qt_trans= ((yhat[t] - y[t-1]).dot(self.w.T))*gt_trans
            q_tau_trans=qt_trans
            for tau in range(t+1,1,-1):
                #print(tau)
                prev_tau_g_trans=(1-(np.tanh(zt[tau-1]))**2).T
                prev_tau_q_trans= (q_tau_trans.dot(self.U))*prev_tau_g_trans
                q_tau_trans=prev_tau_q_trans
                r_tau=q_tau_trans.T.dot(ht[tau-1].T)
                dU+=r_tau
            self.U=self.U-lr*dU
        #print(dU)
        
        
        dV=np.zeros((self.numHidden,self.numInput))
        for t in range(len(zt)-1,0,-1):
            #print(t)
            gt_trans=(1-(np.tanh(zt[t]))**2).T
            qt_trans= ((yhat[t] - y[t-1]).dot(self.w.T))*gt_trans
            q_tau_trans=qt_trans
            for tau in range(t+1,1,-1):
                #print(tau)
                prev_tau_g_trans=(1-(np.tanh(zt[tau-1]))**2).T
                prev_tau_q_trans= (q_tau_trans.dot(self.U))*prev_tau_g_trans
                q_tau_trans=prev_tau_q_trans
                r_tau=q_tau_trans.T.dot(xs[tau-2].T)
                dV+=r_tau
            self.V=self.V-lr*dV
        #print(dV)
        
        
        dw=np.zeros((numHidden,1))
        for i in range(len(yhat)-1,1,-1):  
             #print(ht[i].T.shape)
             dw+=(yhat[i]-y[i-1])*(ht[i])
             self.w=self.w-lr*dw
        #print(dw)

    def forward (self, x,y):
        # TODO: IMPLEMENT ME
        
        #uv = np.hstack((self.U,self.V))
        yhat=[None]
        jt=[0]
        zt=[None]
        h0=np.zeros((numHidden,numInput))
        ht=[h0]
        for i in range(1,len(x)+1):
            #for j in range(1,numHidden+1):
                #if j==1:
                    #hprevx = np.vstack((ht[j-1],x[i-1]))
                #else:
                    #hprevx = np.vstack((hj,x[i-1]))
                #zj=uv.dot(hprevx)
                #hj=np.tanh(zj)
            #print("zj shape",zj.shape)
            zt.append(np.dot(self.U,ht[i-1])+np.dot(self.V,x[i-1]))
            ht.append(np.tanh(zt[i]))
            yhat.append(np.dot(ht[i].T,self.w))
            jt.append(0.05*((yhat[i]-y[i-1])**2))
        #print("zt.shape",zt[2].shape)
        #print(len(jt))
        cost=np.squeeze(sum(jt))
        print('cost',cost)
        return jt,yhat,zt,ht,cost

def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    #print(xs)
    #print (ys)
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)
    for i in range(600):
        print('Epoch',i)
        jt,yhat,zt,ht,cost=rnn.forward(xs,ys)
        if(cost<0.05):
            break
        rnn.backward(yhat,ys,zt,ht,xs)