import numpy as np
import pandas as pd
import pygal.maps.world
def getscore(gdp,water,air,people,k):
    waterweight=0.3218
    airweight=0.6751
    averageGDP=0.358
    peopledense=0.641
    gdp=(gdp-gdp.min(0))/(gdp.max(0)-gdp.min(0))
    water=(water-water.min(0))/(water.max(0)-water.min(0))
    air=(air-air.min(0))/(air.max(0)-air.min(0))
    people=(people-people.min(0))/(people.max(0)-people.min(0))
    return (1/(1+k))*(k*(waterweight*water+air*airweight)+(gdp*averageGDP+people*peopledense))
data=pd.read_csv('./data/SO2.csv')
data=data.dropna(axis=0)
preyear=6
pyear=[i for i in range(2000,2000+preyear*5,5)]
year=np.array(['2000','2005','2010'])
so2set=data.values[:,3:6]
waterusdset=data.values[:,9:12]
solidset=data.values[:,24:27]
wateruwdset=data.values[:,16:19]
popworldset=data.values[:,28:31]
gdpset=data.values[:,33:36]
waterset=(waterusdset+wateruwdset)/2
set1=getscore(gdpset,waterset,so2set,popworldset,1)
datao=pd.DataFrame(set1,data['iso'],columns=['2000','2005','2010'])
datao.to_excel('./data/scorefor2000-2010.xls')
for preyear in range(2000, 2010, 5):
    temp = {}
    for i in datao.index:
        temp.setdefault(i.lower(), datao[str(preyear)][i])
    lanw = pygal.maps.world.World()
    lanw.title = 'World prediction' + str(preyear)
    lanw.add(str(preyear), temp)
    lanw.render_to_file('./picture/' + str(preyear) + 'bar_chart.svg')
