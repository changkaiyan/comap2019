import random
import time
class IndusOntrans:
    def __init__(self,name,a,b,sigma,e):
        self.name=name
        self.e=e
        self.sigma=sigma
        self.z=random.random()
        self.gamma=0.001
        self.q=100
        self.c=0.0
        self.alpha=random.random()
        self.beta=random.random()
        self.d=self.alpha
        self.p=a-b*self.q
        self.s=self.p*self.q-sigma*(self.d*self.q-self.e)-self.c
        self.myinvestment=self.gamma*self.s
    def addCompetitor(self,investment):
        self.d-=self.beta*investment
        self.c+=self.z*investment*investment
        self.s=self.p*self.q-self.sigma*(self.d*self.q-self.e)-self.c
        self.myinvestment=self.gamma*self.s
    def printC(self):
        if(self.myinvestment<=0):
            print('Company'+self.name,':invest will not change.')
        else:
            print('Company'+self.name+':',self.myinvestment)
class IndusClustOntrans:
    def __init__(self,deltaGDP):
        self.deltaGDP=deltaGDP
        self.enterprisenum=random.randint(1,10)+self.deltaGDP
        self.enterprise=[]
        for i in range(self.enterprisenum):
            self.enterprise.append(IndusOntrans(str(i),1000,0.2,11.6,70))
    def runAll(self):
        random.seed(10)
        i=0
        while i<15:
            print('')
            print("The",i,"epoch:")
            f=random.randint(0,self.enterprisenum-1)
            self.printAll()
            for j in self.enterprise:
                if j!=self.enterprise[f]:
                    j.addCompetitor(self.enterprise[f].myinvestment)
            i+=1
    def printAll(self):
        for i in self.enterprise:
            i.printC()
monitor=IndusClustOntrans(20)
monitor.runAll()