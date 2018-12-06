# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:56:03 2018

@author: Lena Aleksandrova
"""

import numpy as np

class mlp2:
    def __init__(self, inputs, targets, nhidden):
        #set up neural networks
        self.beta = 1 
        self.eta = 0.1 #learning rates 
        self.momentum = 0.9 #optional
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin) 
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
      
    def earlystopping(self, inputs, targets, valid, validtargets):
        
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            self.train(inputs,targets,niterations=100)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.forward(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print ("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error

    def train(self, inputs, targets,niterations=100):
       
        #change = list (range(self.ndata))
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)#defining inputs with bias
        change = list (range(self.ndata))
        # variables for weights updating 
        updatewh = np.zeros((np.shape(self.weights1)))
        updatewo = np.zeros((np.shape(self.weights2)))
        #using batch traning method 
        for n in range(niterations):
            self.outputs = self.forward(inputs) # calling forward method 
            # multiplication gradien decent 
            error = ((self.outputs-targets)**2)
            for column in range(len(error[0])):
                total = 0
                for row in range(len(error)):
                    total += error[row][column]
            error1 = 0.5*total
            # print out gradient desent in every iteration
            if ((n%10)==0):
                print ('Epoch:', n , 'Error:',error1)
            deltao = self.outputs-targets # error - different between output and target
            # next for loop compute delta i hidden neurons
            trans_weights2 = np.transpose(self.weights2)
            weight2_deltao = [[0 for g in range(len(trans_weights2[0]))] for r in range(len(deltao))]
            for b in range(len(deltao)):
                for a in range(len(trans_weights2[0])):
                    for q in range(len(trans_weights2)):
                        weight2_deltao[b][a] += deltao[b][q] * trans_weights2[q][a]
            
            deltah1 = self.hidden*self.beta*(1.0-self.hidden)# derivation of input hidden neurons
            deltah = deltah1*weight2_deltao # multiplication of 
            delta_minus1 = deltah[:,:-1] # delated bias   
            self.transpose = np.transpose(inputs) # transpose inputs neurons 
            multipl_wh = [[0 for m in range(len(delta_minus1[0]))] for l in range(len(self.transpose))]
            for o in range(len(self.transpose)):
                for z in range(len(delta_minus1[0])):
                    for w in range(len(delta_minus1)):
                        multipl_wh[o][z] += self.transpose[o][w] * delta_minus1[w][z]
            
            updatewh = self.eta*np.asarray(multipl_wh) + self.momentum*updatewh#  weights in hidden layer
            trans_hidden = np.transpose(self.hidden)# transpose hidden neurons
            multipl_wo = [[0 for s in range(len(deltao[0]))] for v in range(len(trans_hidden))]
            for q in range(len(trans_hidden)):
                for f in range(len(deltao[0])):
                    for j in range(len(deltao)):
                        multipl_wo[q][f] += trans_hidden[q][j] * deltao[j][f]
            
            updatewo = self.eta*np.asarray(multipl_wo) + self.momentum*updatewo#  weights in output layers
            self.weights1 -= updatewh # updating weights in hidden layer 
            self.weights2 -= updatewo # updating weights in output layer
        np.random.shuffle(change) # random change inputs and targets 
        inputs = inputs[change,:]
        targets = targets[change,:]
        
                    
         
    
    def forward(self, inputs):
        # computing inputs and weight in hidden layer for every hidden neurons. Using leanr activation  
        self.hidden = [[0 for x in range(len(self.weights1[0]))] for y in range(len(inputs))]
        for i in range(len(inputs)):
            for j in range(len(self.weights1[0])):
                for k in range(len(self.weights1)):
                    self.hidden[i][j] += inputs[i][k] * self.weights1[k][j]
                if self.hidden[i][j] > 0:
                    self.hidden[i][j] = 1
                else:
                    self.hidden[i][j] = 0
        # 
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = [[0 for n in range(len(self.weights2[0]))] for k in range(len (self.hidden))]
        for p in range (len(self.hidden)):
            for u in range(len(self.weights2[0])):
                for e in range(len(self.weights2)):
                    outputs[p][u] += self.hidden[p][e] * self.weights2[e][u]
                if outputs[p][u] > 0:
                    outputs[p][u] = 1
                else:
                    outputs[p][u] = 0
            
       
        return outputs
                
                         

    def confusion(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.forward(inputs)
        
        nclasses = np.shape(targets)[1]
      

        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print ("Confusion matrix is:" )
        print (cm)
        print ("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)
        #great metrix: correct matrix, see time, how correct your network printing matrix  

