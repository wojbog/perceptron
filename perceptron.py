# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import math


def step(x):
    if x > 0: return 1.0
    else: return -1.0
        
def step_pseudo_derivative(x):
    return 1.0
        
def logistic(x,alpha = 1.0,beta = 1.0,gamma = 2.0):
    return 1.0/(alpha+beta*math.exp(-gamma*x))

def logistic_derivative(x,alpha = 1.0,beta = 1.0,gamma = 2.0):
    return beta*gamma*math.exp(-gamma*x)/((alpha+beta*math.exp(-gamma*x))**2)

class Perceptron:
    '''
    neuron
    '''
    
 
    def dot(self,a,b):#iloczyn skalarny dwóch wektorów
        res = 0
        for i in range(len(a)):
            res += a[i]*b[i]
        return res
    

    def error(self,a,b): #srednia różnica pomiędzy dwoma wektorami
        res = 0.0
        for i in range(len(a)):
            res += abs(a[i]-b[i])
        return res/len(a)
    

    # N - ilosć przejsć algorytmu  uczącego  przez wszystkie dane - to jest wygodne ustawienie ze względu na ogólną teorię zbieżnosci sieci
    # err - sredni błąd uczenia na danych treningowych, przyjęty jako wystarczająco mały
    def __init__(self, learn_speed, N, err, activate, activate_derivative): 
        
        self.learn_speed = learn_speed
        self.N = N 
        self.err = err
        self.activate = activate
        self.activate_derivative = activate_derivative
        
    def learn(self,input,output):  #input - wektor treningowy  - każda jego składowa jest n-wymiarowa, n=2 dla współrzędnych x,y, output - wektor poprawnych klasyfikacji dla input
        
        
        #inicjacja wag perceptronu małymi wartosciami losowymi (z zakresu 0.0-0.2)
        
        self.weights = []
        for i in range(len(input[0])+1):
            self.weights.append(random.random()*0.02) 
        
        indices = []
        for i in range(len(input)): indices.append(i)
        
        for j in range(len(input)):
            input[j].insert(0,1)# dodanie 1 na początku każdego wektora danych
            
        print(input)        
        
        for i in range(self.N): 
            
            print("start epoch: ", i)
        
            random.shuffle(indices)# losowa permutacja indeksów zbioru treningowego
            #print("indices = ", indices)
            
            y = [] # wektor odpowiedzi neuronu na danych treningowych - w jednym przebiegu
            
            for j in range(len(input)):
                y.append(0.0)
            
        
            for j in indices: #losowo ułożone próbki ze zbioru treningowego
                

                #print(j)
                
                y_j = self.activate(self.dot(input[j],self.weights)) # obliczenie odpowiedzi neuronu na j-tej danej  
                
                y[j] = y_j
                
                #print("y_j =", y_j)
                
                for k in range(len(self.weights)): #poprawki wektora wag
                    self.weights[k] +=  self.learn_speed * (output[j] - y_j) * input[j][k]*self.activate_derivative(self.dot(input[j],self.weights))
            
                #print("weights = ", self.weights)
                
            print("error = ", self.error(y,output))
            
            if self.error(y,output)< self.err:
                print("weights = ", self.weights)
                break # rezygnacja z kolejnych przebiegów wobec osiągnięcia pożądanej zbieżnosci
    
        print("weights = ", self.weights)

class Show_binary_classification:
    '''
    params: współczynniki prostej wg równania w*x=0 - iloczyn skalarny    
    '''

    def __init__(self,input,output,params):
        self.params = params
        self.input = input
        self.output = output

        plot_range = [[0.0,0.0], [0.0,0.0]]


        for i in range(len(input)):
            if input[i][1] < plot_range[0][0]:
                plot_range[0][0] = input[i][1]
            else:
                if input[i][1] > plot_range[0][1]:
                    plot_range[0][1] = input[i][1]
        
            if input[i][2] < plot_range[1][0]:
                plot_range[1][0] = input[i][2]
            else:
                if input[i][2] > plot_range[1][1]:
                    plot_range[1][1] = input[i][2]

        self.range = plot_range
        
        
    def plot(self,color, show_axes = True):

        import matplotlib.pyplot as mp
        
        if show_axes:        
            mp.plot(self.range[0], [0,0], color = 'black')
            mp.plot([0,0],self.range[1],color = 'black')
        
        mp.plot(self.range[0],[(-self.params[0]-self.params[1]*self.range[0][0])/self.params[2],(-self.params[0]-self.params[1]*self.range[0][1])/self.params[2]], color = color )
        
        for i in range(len(self.input)):
            if self.output[i] == 1:
                mp.plot(self.input[i][1],self.input[i][2],'o',color='blue')
            else:
                mp.plot(self.input[i][1],self.input[i][2],'o',color='red')
        mp.show()
        
        
#-------------------------------------------------- Test 1 -------------------------------------     
        
data_1 = [[1, 0.5, 1.0], [3, 0.5, 1.0], [2, 5, 1.0], [-3, -1, -1.0], [-2, -2, -1.0], [1, -2, -1.0], [-1, 5, 1.0], [-3, 2, -1.0]]

input = []
output = []

for i in range(len(data_1)):
    input.append(data_1[i][0:2])
    output.append(data_1[i][2])
    

perc_1 = Perceptron(0.5,10,0.01,step,step_pseudo_derivative)
perc_1.learn(input,output)



show_bc = Show_binary_classification( input,output,perc_1.weights)
show_bc.plot('green')

#-------------------------------------------------- Test 2 -------------------------------------     

#niektóre zbiory, które nie są separowalne liniowo generują mimo to optymalne nachylenie - z w miarę najmniejszym średnim błędem - poniżej oczekiwany błąd jest ustawiony na 0.001 

from sklearn.datasets.samples_generator import make_blobs

#X, Y = make_blobs(n_samples=400, centers=2, cluster_std=0.7, random_state=0)
X, Y = make_blobs(n_samples=400, centers=2, cluster_std=0.8, random_state=0)
#X, Y = make_blobs(n_samples=400, centers=2, cluster_std=0.9, random_state=0)


for i in range(len(Y)):
    if Y[i] == 0: Y[i] = -1.0
    else: Y[i] = 1.0


input = []
output = []

for i in range(len(X)):
    input.append([X[i][0],X[i][1]])
    output.append(Y[i])


perc_2 = Perceptron(0.5,1000,0.001,step,step_pseudo_derivative)
perc_2.learn(input,output)



show_bc = Show_binary_classification( input,output,perc_2.weights)
show_bc.plot('green')
