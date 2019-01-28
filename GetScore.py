def getenvscore(water,air,k):
    waterweight=0.3218
    airweight=0.6751
    water=(water-min(water))/(max(water)-min(water))
    air=(air-min(air))/(max(air)-min(air))
    return (waterweight*water+air*airweight)
def geteconoscore(gdp,struct,people,k):
    averageGDP=0.2914
    structure=0.1336
    peopledense=0.5750
    gdp=(gdp-min(gdp))/(max(gdp)-min(gdp))
    struct=(struct-min(struct))/(max(struct)-min(struct))
    people=(people-min(people))/(max(people)-min(people))
    return (gdp*averageGDP+struct*structure+people*peopledense)