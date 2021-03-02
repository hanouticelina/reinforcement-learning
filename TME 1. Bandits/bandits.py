import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import *
import math

# Traitement des donnees
# Comme chaque ligne contient id:context:reward, la classe data permet de bien gérer et stocker les données
class Data:
    def __init__(self, FILENAME):
        self.reward=[] # Contient tous les tableaux de gains
        self.context=[] # contient les tableau de context
        with open(FILENAME) as f:
            for line in f.readlines():
                parts = line.split(':')
                parts[1] = parts[1].split(';')
                parts[2] = parts[2].split(';')
                self.context.append(np.array([float(i) for i in parts[1]]))
                self.reward.append(np.array([float(i) for i in parts[2]]))
        self.reward=np.array(self.reward)
        self.context=np.array(self.context)
    
    def __iter__(self):
        # à chaque fois on traite un tuple context,reward
        for i in range(len(self.reward)):
            yield self.context[i], self.reward[i]
            
            
class RandomStrategy(): 
    def __init__(self):
        pass
    def step(self,context,reward):
        # permet de jouer un coup, retourne l'index du annonceur choisit
        rand_idx = np.random.randint(low = 0, high = len(reward))
        return rand_idx
    
    
    
# la classe permet de jouer en suivant la stratégie StaticBest à chaque fois
class StaticBestStrategy():
    def __init__(self, data):
        mu = np.array(list(data))[:,1] # contient la moyenne de gain général pour chaque annonceur
        self.best = np.argmax(np.sum(mu, axis = 0)) # l'index de l'annonceur qui a la meilleure moyenne géneral
    def step(self,context,reward):
        # permet de jouer un coup, retourne l'index du annonceur choisit
        return self.best
    
class OptimaleStrategy :
    def __init__(self):
        pass

    def step(self,context,reward):  
        # permet de jouer un coup, retourne l'index du annonceur choisit
        return np.argmax(np.array(reward))
    
    
# La classe abstraite permet de définir le squelette de toutes stratégie
class Bandit:
    def __init__(self):
        pass
    def step(self,context,reward):           
    # permet de jouer un coup, retourne l'index du annonceur choisit
        action = self.action(context)
        self.update(reward,action)
        return action
    def action(self,context):
        pass
    def update(self,reward,action):
        pass

# La classe abstraite permet de définir le squelette des stratégies qui comptent les annonceurs choisis dans le passé pour choisir l'annonceur
class CountBased(Bandit):
    def __init__(self):
        pass
    def update(self,reward,action):
         # permet de jouer un coup (trouver l'action selon la stratégie et mettre à jour les parametres), retourne l'index du annonceur choisit
        self.mu[action]=(self.mu[action]*self.nb[action]+reward[action])/(self.nb[action]+1) # mu = ((mu*s) + gain)/(s+1) avec mu l'ancien moyenne et s le nombre de fois qu'on a utilisé cet annonceur
        self.nb[action]+=1
        
# la classe permet de jouer en suivant la stratégie UCB à chaque fois
class UCB(CountBased):
    def __init__(self,data):
        #initialisation avec les 10 premiers tuples (context,reward) pour chaque tuple i on utilise l'annonceur i 
        self.mu = np.stack(np.array(list(data))[:10,1]).diagonal()
        # le nombre de fois ou on a utilisé les annonceurs
        self.nb=np.ones((list(data)[0][1].shape[0]))
        self.mu.flags.writeable = True
    def action(self,context):
        # permet de choisir la bonne action selon la stratégie 
        return np.argmax(self.mu+np.power(2*np.log(np.sum(self.nb))/self.nb,1/2))
    
    
# la classe permet de jouer en suivant la stratégie UCB à chaque fois
class E_Greedy(CountBased):
    def __init__(self,data,e=0.1):
        #initialisation avec les 10 premiers tuples (context,reward) pour chaque tuple i on utilise l'annonceur i 
        self.mu = np.stack(np.array(list(data))[:10,1]).diagonal()
         # le nombre de fois ou on a utilisé les annonceurs
        self.nb=np.ones((list(data)[0][1].shape[0]))
        # learning rate
        self.e=e
        self.mu.flags.writeable = True
    def action(self,context):
         # permet de choisir la bonne action selon la stratégie 
        a = random()
        if(a<self.e):
            return np.random.randint(low = 0, high = len(context))
        return np.argmax(self.mu)
    
# la classe permet de jouer en suivant la stratégie UCB à chaque fois
class LinUCB(Bandit):
    
    def __init__(self,data,alpha=0.2):
        # le nombre d'annonceurs
        self.nb=list(data)[0][1].shape[0]
        # coeff d'éxploration
        self.alpha=alpha
        # la dimention de context
        self.d =list(data)[0][0].shape[0]
        self.A=[np.identity(self.d)]*self.nb
        self.b=np.zeros((self.nb,self.d))

    def step(self,context,reward):
        # permet de jouer un coup (trouver l'action selon la stratégie et mettre à jour les parametres), retourne l'index du annonceur choisit
        context=context.reshape((self.d,1))
        action = self.action(context)
        self.update(action,reward,context)
        return action
    
    def action(self,context):
        # permet de choisir la bonne action selon la stratégie 
        val=np.zeros(self.nb)
        for i in range(self.nb):
            teta=np.dot(np.linalg.inv(self.A[i]),self.b[i]).reshape((self.d,1))
            p=np.dot(teta.T,context)+self.alpha*np.sqrt(np.dot(np.dot(context.T,np.linalg.inv(self.A[i])),context))    
            val[i]=p 
        return np.random.choice(np.where(val == val.max())[0])
    
    def update(self, action,reward,context):
        self.A[action]+=np.dot(context,context.T)
        self.b[action]+=reward[action]*context[:,0]