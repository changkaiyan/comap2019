from pandas import Series
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt


class Gray_model:
    def __init__(self):
        self.a_hat = None
        self.x0 = None

    def fit(self,index, data):
        """
        Series is a pd.Series with index as its date.
        :param series: pd.Series
        :return: None
        """
        series=pd.Series(index=index, data=data)
        self.a_hat = self._identification_algorithm(series.values)
        self.x0 = series.values[0]

    def predict(self, interval):
        result = []
        for i in range(interval):
            result.append(self.__compute(i))
        tmp = np.ones(len(result))
        for i in range(len(result)):
            if i == 0:
                tmp[i] = result[i]
            else:
                tmp[i] = result[i] - result[i - 1]
        result = tem
        return result

    def _identification_algorithm(self, series):
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

    def score(self, series_true, series_pred, index):
        error = np.ones(len(series_true))
        relativeError = np.ones(len(series_true))
        for i in range(len(series_true)):
            error[i] = series_true[i] - series_pred[i]
            relativeError[i] = error[i] / series_pred[i] * 100
        score_record = {'GM': np.cumsum(series_pred),
                        '1—AGO': np.cumsum(series_true),
                        'Returnvalue': series_pred,
                        'Real_value': series_true,
                        'Error': error,
                        'RelativeError(%)': (relativeError)
                        }
        scores = DataFrame(score_record, index=index)
        return scores

    def __compute(self, k):
        return (self.x0 - self.a_hat[1] / self.a_hat[0]) * np.exp(-1 * self.a_hat[0] * k) + self.a_hat[1] / self.a_hat[
            0]

    def evaluate(self, series_true, series_pred):
        scores = self.score(series_true, series_pred, np.arange(len(series_true)))

        error_square = np.dot(scores, np.transpose(scores))
        error_avg = np.mean(error_square)

        S = 0
        for i in range(1, len(series_true) - 1, 1):
            S += series_true[i] - series_true[0] + (series_pred[-1] - series_pred[0]) / 2
        S = np.abs(S)

        SK = 0
        for i in range(1, len(series_true) - 1, 1):
            SK += series_pred[i] - series_pred[0] + (series_pred[-1] - series_pred[0]) / 2
        SK = np.abs(SK)

        S_Sub = 0
        for i in range(1, len(series_true) - 1, 1):
            S_Sub += series_true[i] - series_true[0] - (series_pred[i] - series_pred[0]) + ((series_true[-1] -
                                                                                             series_true[0]) - (
                                                                                            series_pred[i] -
                                                                                            series_pred[0])) / 2
        S_Sub = np.abs(S_Sub)

        acc = (1 + S + SK) / (1 + S + SK + S_Sub)

        level = 0
        if acc >= 0.9:
            level = 1
        elif acc >= 0.8:
            level = 2
        elif acc >= 0.7:
            level = 3
        elif acc >= 0.6:
            level = 4
        return 1 - acc, level

    def plot(self, series_true, series_pred, index,name):
        df = pd.DataFrame(index=index)
        df['Real'] = series_true
        df['Forcast'] = series_pred
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
shanghai=pd.read_excel('./data/huidu.xlsx',index_col=0)#07-14
dataset=shanghai.values
gdpmodel=Gray_model()
airmodel=Gray_model()
watermodel=Gray_model()
strucmodel=Gray_model()
densemodel=Gray_model()
predictyear=2007#初始年份
predictnum=30#预测年度
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
