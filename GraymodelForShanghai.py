import pandas as pd
import matplotlib.pyplot as plt


class GM:
    def __init__(self):
        self.dat = None
        self.x0 = None

    def fit(self,index, data):
        series=pd.Series(index=index, data=data)
        self.dat = self.__identification_algorithm(series.values)
        self.x0 = series.values[0]

    def predict(self, yearnum):
        result = []
        for i in range(yearnum):
            result.append(self.__calcugray(i))
        temp = np.ones(len(result))
        for i in range(len(result)):
            if i == 0:
                temp[i] = result[i]
            else:
                temp[i] = result[i] - result[i - 1]
        result = temp
        return result

    def __identification_algorithm(self, series):
        B = np.array([[1] * 2] * (len(series) - 1))
        series_sum = np.cumsum(series)
        for i in range(len(series) - 1):
            B[i][0] = (series_sum[i] + series_sum[i + 1]) * (-1.0) / 2
        Y = np.transpose(series[1:])
        BT = np.transpose(B)
        a = np.linalg.inv(np.dot(BT, B))
        a = np.dot(a, BT)
        a = np.dot(a, Y)
        a = np.transpose(a)
        return a

    def score(self, series_true, series_prediction, index):
        error = np.ones(len(series_true))
        relativeError = np.ones(len(series_true))
        for i in range(len(series_true)):
            error[i] = series_true[i] - series_prediction[i]
            relativeError[i] = error[i] / series_prediction[i] * 100
        score_record = {'GM': np.cumsum(series_prediction),
                        '1—AGO': np.cumsum(series_true),
                        'Returnvalue': series_prediction,
                        'Real_value': series_true,
                        'Error': error,
                        'RelativeError(%)': (relativeError)
                        }
        scores = pd.DataFrame(score_record, index=index)
        return scores

    def __calcugray(self, k):
        return (self.x0 - self.dat[1] / self.dat[0]) * np.exp(-1 * self.dat[0] * k) + self.dat[1] / self.dat[
            0]

    def evaluate(self, series_true, series_prediction):

        S = 0
        for i in range(1, len(series_true) - 1, 1):
            S += series_true[i] - series_true[0] + (series_prediction[-1] - series_prediction[0]) / 2
        S = np.abs(S)

        SK = 0
        for i in range(1, len(series_true) - 1, 1):
            SK += series_prediction[i] - series_prediction[0] + (series_prediction[-1] - series_prediction[0]) / 2
        SK = np.abs(SK)

        S_Sub = 0
        for i in range(1, len(series_true) - 1, 1):
            S_Sub += series_true[i] - series_true[0] - (series_prediction[i] - series_prediction[0]) + ((series_true[-1] -
                                                                                             series_true[0]) - (
                                                                                            series_prediction[i] -
                                                                                            series_prediction[0])) / 2
        S_Sub = np.abs(S_Sub)

        acc = (1 + S + SK) / (1 + S + SK + S_Sub)

        level = -1
        if acc >= 0.9:
            level = 1
        elif acc >= 0.8:
            level = 2
        elif acc >= 0.7:
            level = 3
        elif acc >= 0.6:
            level = 4
        return 1 - acc, level

    def plot(self, series_true, series_prediction, index,name):
        df = pd.DataFrame(index=index)
        df['Real'] = series_true
        df['Prediction'] = series_prediction
        plt.figure()
        df.plot(figsize=(7, 5))
        plt.xlabel('year')
        plt.title(name)
        plt.show()
def getscore(gdp,water,air,struct,people,k):
    waterweight=0.3218
    airweight=0.6751
    averageGDP=0.2914
    structure=0.1336
    peopledense=0.5750
    gdp=(gdp-min(gdp))/(max(gdp)-min(gdp))
    water=(water-min(water))/(max(water)-min(water))
    air=(air-min(air))/(max(air)-min(air))
    struct=(struct-min(struct))/(max(struct)-min(struct))
    people=(people-min(people))/(max(people)-min(people))
    return (1/(1+k))*(k*(waterweight*water+air*airweight)+(gdp*averageGDP+struct*structure+people*peopledense))
shanghai=pd.read_excel('./data/huidu.xlsx',index_col=0)
dataset=shanghai.values
gdpmodel=GM()
airmodel=GM()
watermodel=GM()
strucmodel=GM()
densemodel=GM()
predictyear=2007
predictnum=30
year=dataset[:,0]
gdpmodel.fit(index=year,data=dataset[:,1])
valyear=[i for i in range(2007,2015)]
error=[]
airmodel.fit(index=year,data=dataset[:,3])
watermodel.fit(index=year,data=dataset[:,2])
strucmodel.fit(index=year,data=dataset[:,-2])
densemodel.fit(index=year,data=dataset[:,-1])
year=[i for i in range(predictyear,predictyear+predictnum)]
gdparray=gdpmodel.predict(predictnum)

errorgdp=gdpmodel.evaluate(dataset[:8,1],gdparray[:8])
gdpmodel.plot(dataset[:8,1],gdparray[:8],valyear,'GDP')
error.append(errorgdp)
mydata=pd.DataFrame(gdparray,year,columns=['GDP'])
airarray=airmodel.predict(predictnum)
mydata=mydata.join(pd.DataFrame(airarray,year,columns=['Air']))
waterarray=watermodel.predict(predictnum)


errorair=airmodel.evaluate(dataset[:8,3],airarray[:8])
airmodel.plot(dataset[:8,3],airarray[:8],valyear,'Air')
errorwater=watermodel.evaluate(dataset[:8,2],waterarray[:8])
watermodel.plot(dataset[:8,2],waterarray[:8],valyear,'Water')
error.append(errorair)
error.append(errorwater)

mydata=mydata.join(pd.DataFrame(waterarray,year,columns=['Water']))
strucarray=strucmodel.predict(predictnum)
mydata=mydata.join(pd.DataFrame(strucarray,year,columns=['Struct']))
densearray=densemodel.predict(predictnum)
mydata=mydata.join(pd.DataFrame(densearray,year,columns=['Dense']))
score=getscore(gdparray,waterarray,airarray,strucarray,densearray,1)
mydata=mydata.join(pd.DataFrame(score,year,columns=['Score']))

errorstruct=strucmodel.evaluate(dataset[:8,-2],strucarray[:8])
strucmodel.plot(dataset[:8,-2],strucarray[:8],valyear,'Structure')
errordense=densemodel.evaluate(dataset[:8,-1],densearray[:8])
densemodel.plot(dataset[:8,-1],densearray[:8],valyear,'Dense')

error.append(errorstruct)
error.append(errordense)
print(error)
plt.plot(year,score)
plt.xlabel('Year')
plt.ylabel('Score')
plt.title('Shanghai prediction for the environment score')
plt.savefig('./picture/shanghaipredict.png')
mydata.to_excel('./data/Shanghaipredict.xls')
