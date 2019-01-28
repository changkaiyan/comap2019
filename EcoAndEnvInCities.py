from GetScore import *
data=pd.read_excel('./data/citytocheck.xlsx',index_col=0)
dataset=data.values
ran=np.linspace(0.6,1.4,9)
a=getenvscore(data['averagePollution'].values[0,0],data['averageSO2'].values[0,0],1)
b=geteconoscore(data['averageGDP'].values,data['structure'].values,data['dense'].values,1)
plt.scatter(a,b,label=list(data.index))
plt.xlabel('environment score')
plt.ylabel('economy score')
plt.legend()
plt.savefig('./picture/1.png')
plt.title('Relationship between Ecomomic and Environment')
plt.savefig('./picture/relationship.png')