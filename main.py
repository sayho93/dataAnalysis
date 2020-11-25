import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('muted')
sns.set_style('whitegrid')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# -------흡연율
smokingData = pd.read_excel('./smokingRate.xlsx')

smokingByAge = smokingData.iloc[3:11]
smokingByAge = smokingByAge.drop(['성별(1)', '응답자특성별(1)'], axis=1)
smokingByAge = smokingByAge[
    ['응답자특성별(2)', '1998.1', '2001.1', '2005.1', '2007.1', '2008.1', '2009.1', '2010.1', '2011.1', '2012.1', '2013.1',
     '2014.1', '2015.1', '2016.1', '2017.1', '2018.1']]
index = smokingByAge.iloc[:, 0].values
smokingByAge.index = index
smokingByAge = smokingByAge.drop(['응답자특성별(2)'], axis=1)
smokingByAge.columns = ['1998', '2001', '2005', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015',
                        '2016',
                        '2017', '2018']
print(smokingByAge)

for idx, row in smokingByAge.iterrows():
    plt.plot(smokingByAge.columns, smokingByAge.loc[idx].to_numpy(), marker='o')

plt.title("흡연률", fontsize=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('%', fontsize=14)
plt.legend(smokingByAge.index)
# plt.show()
plt.savefig("res/smokingRate.png", bbox_inches='tight')
plt.close()


# -------음주율
drinkingData = pd.read_excel("./drinkingRate.xlsx")
drinkingByAge = drinkingData.iloc[3:11]
drinkingByAge = drinkingByAge.drop(['성별(1)', '응답자특성별(1)'], axis=1)
drinkingByAge = drinkingByAge[
    ['응답자특성별(2)', '2005.1', '2007.1', '2008.1', '2009.1', '2010.1', '2011.1', '2012.1', '2013.1', '2014.1', '2015.1',
     '2016.1', '2017.1', '2018.1']]

index = drinkingByAge.iloc[:, 0].values
drinkingByAge.index = index

print(drinkingByAge)
drinkingByAge = drinkingByAge.drop(['응답자특성별(2)'], axis=1)
drinkingByAge.columns = ['2005', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
                         '2018']
print(drinkingByAge)

for idx, row in drinkingByAge.iterrows():
    plt.plot(drinkingByAge.columns, drinkingByAge.loc[idx].to_numpy(), marker='o')

plt.title("음주율", fontsize=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('%', fontsize=14)
plt.legend(drinkingByAge.index)
# plt.show()
plt.savefig("res/drinkingRate.png", bbox_inches='tight')
plt.close()

# -------Cancer
cancerData = pd.read_excel('./cancer.xlsx')

mask = (cancerData.성별 == "남자") & (cancerData.항목 == "연령군발생률")
mask2 = (cancerData.성별 == "여자") & (cancerData.항목 == "연령군발생률")
cancerDataMale = cancerData.loc[mask, :]
cancerDataFemale = cancerData.loc[mask2, :]

cancerDataMale = cancerDataMale.fillna(0)
cancerDataFemale = cancerDataFemale.fillna(0)

cancerDataMale = cancerDataMale.replace('-', 0)
cancerDataFemale = cancerDataFemale.replace('-', 0)

mask = cancerDataMale.연령군 == "계"
mask2 = cancerDataFemale.연령군 == "계"
cancerDataMale = cancerDataMale.loc[mask, :]
cancerDataFemale = cancerDataFemale.loc[mask2, :]

masked = cancerDataMale
masked = masked.drop(['성별', '연령군', '항목', '단위'], axis=1)
masked = masked.reset_index(drop=True)
maskedFemale = cancerDataFemale
maskedFemale = maskedFemale.drop(['성별', '연령군', '항목', '단위'], axis=1)
maskedFemale = maskedFemale.reset_index(drop=True)


def plotByYear(data, i, gender):
    txt = gender == 1 and "남자" or "여자"
    tmp = data.iloc[[i]]
    title = tmp.iloc[0, 0] + " - " + txt
    tmp = tmp.transpose()
    tmp = tmp.drop(['24개 암종'])
    print(title)
    ax = sns.barplot(
        data=tmp,
        x=tmp.index,
        y=tmp.iloc[:, 0]
    )
    ax.set_title(title)
    plt.savefig("res/type/type{}-{}.png".format(i, txt), bbox_inches='tight')
    plt.close()


def plotByDisease(data, index, gender):
    txt = gender == 1 and "남자" or "여자"
    title = data.index.values[index] + " - " + txt
    print(title)

    print(data.iloc[index][1:])
    ax = sns.barplot(
        data=data,
        x=data.iloc[index][1:],
        y=data.iloc[0][1:]
    )
    ax.set_title(title)
    plt.savefig("res/year/year{}-{}.png".format(i, txt), bbox_inches='tight')
    plt.close()


plt.rcParams["figure.figsize"] = (14, 4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

for i in range(0, 24):
    plotByYear(masked, i, 1)
    plotByYear(maskedFemale, i, 0)


plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams['lines.linewidth'] = 0.1
plt.rcParams['axes.grid'] = True

tmp = masked.transpose()
tmp2 = maskedFemale.transpose()

for i in range(1, 20):
    plotByDisease(tmp, i, 1)
    plotByDisease(tmp2, i, 0)

