import random
import time
class Indus:
    def __init__(self,name):
        self.name=name
        self.z=random.random()
        self.myinvestment=random.random()*100
        self.c=0.0
    def addCompetitor(self,investment):
        self.c=self.z*investment*investment
        self.myinvestment+=self.c
    def printC(self):
        print('Company '+self.name+':',self.myinvestment)
class IndusClust:
    def __init__(self,deltaGDP):
        self.deltaGDP=deltaGDP
        self.enterprisenum=random.randint(1,10)+self.deltaGDP
        self.enterprise=[]
        for i in range(self.enterprisenum):
            self.enterprise.append(Indus(str(i)))
    def runAll(self):
        random.seed(10)
        i=0
        while i<100:
            print('')
            print("The ",i,"epoch:")
            f=random.randint(0,self.enterprisenum-1)
            #self.enterprise[f].printC()
            self.printAll()
            for j in self.enterprise:
                if j!=self.enterprise[f]:
                    j.addCompetitor(self.enterprise[f].myinvestment)
            i+=1
            time.sleep(1)
    def printAll(self):
        for i in self.enterprise:
            i.printC()
monitor=IndusClust(20)
monitor.runAll()