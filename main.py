import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.formula.api import ols

sns.set_palette('muted')
sns.set_style('whitegrid')
pd.set_option('display.max_rows', 3200)
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
# print(smokingByAge)

# for idx, row in smokingByAge.iterrows():
#     plt.plot(smokingByAge.columns, smokingByAge.loc[idx].to_numpy(), marker='o')
#
# plt.title("흡연률", fontsize=20)
# plt.xlabel('Year', fontsize=14)
# plt.ylabel('%', fontsize=14)
# plt.legend(smokingByAge.index)
# # plt.show()
# plt.savefig("res/smokingRate.png", bbox_inches='tight')
# plt.close()


# -------음주율
drinkingData = pd.read_excel("./drinkingRate.xlsx")
drinkingByAge = drinkingData.iloc[3:11]
drinkingByAge = drinkingByAge.drop(['성별(1)', '응답자특성별(1)'], axis=1)
drinkingByAge = drinkingByAge[
    ['응답자특성별(2)', '2005.1', '2007.1', '2008.1', '2009.1', '2010.1', '2011.1', '2012.1', '2013.1', '2014.1', '2015.1',
     '2016.1', '2017.1', '2018.1']]

index = drinkingByAge.iloc[:, 0].values
drinkingByAge.index = index

# print(drinkingByAge)
drinkingByAge = drinkingByAge.drop(['응답자특성별(2)'], axis=1)
drinkingByAge.columns = ['2005', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
                         '2018']
# print(drinkingByAge)

# for idx, row in drinkingByAge.iterrows():
#     plt.plot(drinkingByAge.columns, drinkingByAge.loc[idx].to_numpy(), marker='o')
#
# plt.title("음주율", fontsize=20)
# plt.xlabel('Year', fontsize=14)
# plt.ylabel('%', fontsize=14)
# plt.legend(drinkingByAge.index)
# # plt.show()
# plt.savefig("res/drinkingRate.png", bbox_inches='tight')
# plt.close()

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
    plt.savefig("res/year/year{}-{}.png".format(index, txt), bbox_inches='tight')
    plt.close()


# plt.rcParams["figure.figsize"] = (14, 4)
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['axes.grid'] = True
#
# for i in range(0, 24):
#     plotByYear(masked, i, 1)
#     plotByYear(maskedFemale, i, 0)
#
#
# plt.rcParams["figure.figsize"] = (10, 10)
# plt.rcParams['lines.linewidth'] = 0.1
# plt.rcParams['axes.grid'] = True
#
# tmp = masked.transpose()
# tmp2 = maskedFemale.transpose()
#
# for i in range(1, 20):
#     plotByDisease(tmp, i, 1)
#     plotByDisease(tmp2, i, 0)

data = pd.read_csv('./data.csv', error_bad_lines=False, encoding="CP949")
data = data.drop('Unnamed: 0', axis=1)

"""
DI1_dg - 고혈압 의사진단 여부 0: 없음 1: 있음 8: 비해당(청소년, 소아) 9: 모름, 무응답
DI1_ag - 고혈압 진단시기 0~79: 0~79세 80: 80세이상 888:비해당(청소년, 소아, 의사진단 받지 않음) 999: 모름, 무응답
DI1_pr - 고혈압 현재 유병 여부 0: 없음 1: 있음 8: 비해당 9: 모름, 무응답
DI1_pt - 고혈압 치료 여부 0: 없음 1: 있음 8: 비해당 9: 모름, 무응답
DI1_2 - 혈압조절제 복용 1: 매일 목용 2: 한달에 20일 이상 3: 한달에 15일 이상 4: 한달에 15일 미만 5: 미복용 8: 비해당 9: 모름, 무응답

BD1 - (만12세이상)평생음주경험
BD2 - (만12세이상)음주 시작 연령
BD1_11 - (만12세이상)1년간 음주빈도
BD2_1 - (만12세이상)한번에 마시는 음주량
BD2_14 - (만12세이상)한번에 마시는 음주량_잔
BD2_31 - (만12세이상) 폭음 빈도
BD2_32 - (성인) 여자 폭음 빈도

BS1_1 - (성인) 평생흡연 여부
BS2_1 - (성인) 흡연 시작연령
BS2_2 - (성인)매일 흡연 시작연령
BS3_1 - (성인) 현재흡연 여부
BS3_2 - (성인) 하루평균 흡연량
BS3_3 - (성인) 가끔흡연자 최근 1달간 흡연일수
BS6_2 - (성인) 과거흡연자 흡연기간(월 환산)
BS6_2_1 - (성인) 과거흡연자 흡연기간(연)
BS6_2_2 - (성인) 과거흡연자 흡연기간(월)
BS6_3 - (성인) 과거흡연자 하루평균 흡연량
BS6_4 - (성인) 과거흡연자 금연기간(월 환산)
BS6_4_1 - (성인) 과거흡연자 금연기간(연)
BS6_4_2 - (성인) 과거흡연자 금연기간(월)
"""

params = ["age", "DI1_dg", "DI1_ag", "DI1_pr", "DI1_pt", "DI1_2", "BD1", "BD2", "BD1_11", "BD2_1", "BD2_14", "BD2_31",
          "BD2_32"]
highBP = data[params]
# mask = highBP.DI1_dg == 1
# highBP = highBP.loc[mask, :]
# print(highBP)
# print(highBP.count())

# mask = (highBP.BD1_11 >= 4) & (highBP.BD1_11 != 8) & (highBP.BD1_11 != 9)
mask = (highBP.DI1_ag != 999) & (highBP.DI1_ag != 888)
highBP = highBP.loc[mask, :]
# print(highBP)
# print(highBP.count())
plt.scatter(highBP["BD2_31"].values, highBP["DI1_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title('폭음 빈도 vs 고혈압 진단시기')
plt.xlabel('BD2_31')
plt.ylabel('DI1_ag')
plt.grid()
# plt.show()
plt.savefig("res/폭음빈도 vs 고혈압 진단시기.png", bbox_inches='tight')
plt.close()

params = ["age", "DI1_dg", "DI1_ag", "DI1_pr", "DI1_pt", "DI1_2", "BS1_1", "BS2_1", "BS2_2", "BS3_1", "BS3_2", "BS3_3",
          "BS6_2", "BS6_2_1", "BS6_2_2", "BS6_3", "BS6_4", "BS6_4_1", "BS6_4_2"]
highBP = data[params]

mask = (highBP.BS2_2 != 999) & (highBP.BS2_2 != 888) & (highBP.DI1_ag != 999) & (highBP.DI1_ag != 888) & \
       (highBP.DI1_dg == 1)
highBP = highBP.loc[mask, :]
# print(highBP)
title = "고혈압 진단시기 vs 매입흡연 시작연령"
plt.scatter(highBP["BS2_2"].values, highBP["DI1_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('BS2_2')
plt.ylabel('DI1_ag')
plt.grid()
# plt.show()
plt.savefig("res/고혈압 진단시기 vs 매일흡연 시작연령.png", bbox_inches='tight')
plt.close()

X = highBP[["BS2_2"]]
Y = highBP[["DI1_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("BS2_2")
plt.ylabel("DI1_ag")
plt.grid()
# plt.show()
plt.savefig("res/고혈압 진단시기 vs 매일흡연 시작연령_regression.png", bbox_inches='tight')
plt.close()
print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DI1_ag~BS2_2', data=highBP).fit()
print(title)
print(res.summary())

# ------------------------------------ Age - highBP
params = ["age", "DI1_dg", "DI1_ag", "DI1_pr", "DI1_pt", "DI1_2", "BS1_1", "BS2_1", "BS2_2", "BS3_1", "BS3_2", "BS3_3",
          "BS6_2", "BS6_2_1", "BS6_2_2", "BS6_3", "BS6_4", "BS6_4_1", "BS6_4_2"]
highBP = data[params]

mask = (highBP.DI1_ag != 999) & (highBP.DI1_ag != 888) & (highBP.DI1_dg == 1)
highBP = highBP.loc[mask, :]

plt.scatter(highBP["age"].values, highBP["DI1_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title('고혈압 진단시기 vs 연령')
plt.xlabel('age')
plt.ylabel('DI1_ag')
plt.grid()
# plt.show()
plt.savefig("res/고혈압 진단시기 vs 연령.png", bbox_inches='tight')
plt.close()

X = highBP[["age"]]
Y = highBP[["DI1_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)

plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title("고혈압 진단시기 vs 연령")
plt.xlabel("연령")
plt.ylabel("고혈압 진단시기")
plt.grid()
plt.savefig("res/고혈압 진단시기 vs 연령_regression.png", bbox_inches='tight')
# plt.show()
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DI1_ag~age', data=highBP).fit()
print(res.summary())

# --------------------------------- gastric cancer
params = ["age", "DC1_dg", "DC1_ag", "DC1_pr", "DC1_pt", "DI1_2", "BS1_1", "BS2_1", "BS2_2", "BS3_1", "BS3_2", "BS3_3",
          "BS6_2", "BS6_2_1", "BS6_2_2", "BS6_3", "BS6_4", "BS6_4_1", "BS6_4_2", "BD1", "BD2", "BD1_11", "BD2_1",
          "BD2_14", "BD2_31", "BD2_32"]
gastricCancer = data[params]
mask = (gastricCancer.BD2 != 999) & (gastricCancer.BD2 != 888) & (gastricCancer.DC1_dg == 1)
gastricCancer = gastricCancer.loc[mask, :]
print(gastricCancer)

plt.scatter(gastricCancer["DC1_ag"].values, gastricCancer["BD2"].values, c="steelblue", edgecolor="white", s=20)
plt.title('음주 시작 연령 vs 위암 진단시기')
plt.xlabel('DC1_ag')
plt.ylabel('BD2')
plt.grid()
# plt.show()
plt.savefig("res/음주 시작 연령 vs 위암 진단시기.png", bbox_inches='tight')
plt.close()

# --------------------------------- highBP income

params = ["age", "ainc", "DI1_dg", "DI1_ag", "DI1_pr", "DI1_pt", "DI1_2", "BS1_1", "BS2_1", "BS2_2", "BS3_1", "BS3_2",
          "BS3_3", "BS6_2", "BS6_2_1", "BS6_2_2", "BS6_3", "BS6_4", "BS6_4_1", "BS6_4_2"]
highBP = data[params]
highBP = highBP[highBP['ainc'].notna()]
mask = (highBP.DI1_ag != 999) & (highBP.DI1_ag != 888) & (highBP.DI1_dg == 1)
highBP = highBP.loc[mask, :]
title = "고혈압 진단시기 vs 월평균 가구총소득"
plt.scatter(highBP["ainc"].values, highBP["DI1_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('ainc')
plt.ylabel('DI1_ag')
plt.grid()
# plt.show()
plt.savefig("res/고혈압 진단시기 vs 월평균 가구총소득.png", bbox_inches='tight')
plt.close()

X = highBP[["ainc"]]
Y = highBP[["DI1_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("BS2_2")
plt.ylabel("DI1_ag")
plt.grid()
# plt.show()
plt.savefig("res/고혈압 진단시기 vs 월평균 가구총소득_regression.png", bbox_inches='tight')
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DI1_ag~ainc', data=highBP).fit()
print(title)
print(res.summary())

# --------------------------------- stroke / drinking
params = ["age", "ainc", "DI3_dg", "DI3_ag", "DI3_pr", "DI3_pt", "BS1_1", "BS2_1", "BS2_2", "BS3_1", "BS3_2",
          "BS3_3", "BS6_2", "BS6_2_1", "BS6_2_2", "BS6_3", "BS6_4", "BS6_4_1", "BS6_4_2", "BD1", "BD2", "BD1_11",
          "BD2_1", "BD2_14", "BD2_31", "BD2_32"]
stroke = data[params]

mask = (stroke.DI3_ag != 999) & (stroke.DI3_ag != 888) & (stroke.DI3_dg == 1) & (stroke.BD2 != 888) & (stroke.BD2 != 999)
stroke = stroke.loc[mask, :]
title = "뇌졸중 진단시기 vs 음주 시작 연령"
plt.scatter(stroke["BD2"].values, stroke["DI3_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('음주 시작 연령')
plt.ylabel('뇌졸중 진단시기')
plt.grid()
# plt.show()
plt.savefig("res/뇌졸중 진단시기 vs 음주 시작 연령.png", bbox_inches='tight')
plt.close()

X = stroke[["BD2"]]
Y = stroke[["DI3_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("음주 시작 연령")
plt.ylabel("뇌졸중 진단시기")
plt.grid()
# plt.show()
plt.savefig("res/뇌졸중 진단시기 vs 음주 시작 연령_regression.png", bbox_inches='tight')
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DI3_ag~BD2', data=stroke).fit()
print(title)
print(res.summary())

# --------------------------------- stroke / smoking

params = ["age", "ainc", "DI3_dg", "DI3_ag", "DI3_pr", "DI3_pt", "BS1_1", "BS2_1", "BS2_2", "BS3_1", "BS3_2",
          "BS3_3", "BS6_2", "BS6_2_1", "BS6_2_2", "BS6_3", "BS6_4", "BS6_4_1", "BS6_4_2", "BD1", "BD2", "BD1_11",
          "BD2_1", "BD2_14", "BD2_31", "BD2_32"]
stroke = data[params]

mask = (stroke.DI3_ag != 999) & (stroke.DI3_ag != 888) & (stroke.DI3_dg == 1) & (stroke.BS2_2 != 888) & (stroke.BS2_2 != 999)
stroke = stroke.loc[mask, :]
title = '뇌졸중 진단시기 vs 흡연 시작 연령'
plt.scatter(stroke["BS2_2"].values, stroke["DI3_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('흡연 시작 연령')
plt.ylabel('뇌졸중 진단시기')
plt.grid()
# plt.show()
plt.savefig("res/뇌졸중 진단시기 vs 흡연 시작 연령.png", bbox_inches='tight')
plt.close()

X = stroke[["BS2_2"]]
Y = stroke[["DI3_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("흡연 시작 연령")
plt.ylabel("뇌졸중 진단시기")
plt.grid()
# plt.show()
plt.savefig("res/뇌졸중 진단시기 vs 흡연 시작 연령_regression.png", bbox_inches='tight')
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DI3_ag~BS2_2', data=stroke).fit()
print(title)
print(res.summary())


# --------------------------------- stroke / chol

params = ["age", "ainc", "DI3_dg", "DI3_ag", "DI3_pr", "DI3_pt", "HE_chol", "HE_TG"]
stroke = data[params]

mask = (stroke.DI3_ag != 999) & (stroke.DI3_ag != 888) & (stroke.DI3_dg == 1)
stroke = stroke.loc[mask, :]
stroke = stroke[stroke['HE_chol'].notna()]
title = '뇌졸중 진단시기 vs 혈중 콜레스테롤'
plt.scatter(stroke["HE_chol"].values, stroke["DI3_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('혈중 콜레스테롤')
plt.ylabel('뇌졸중 진단시기')
plt.grid()
# plt.show()
plt.savefig("res/뇌졸중 진단시기 vs 혈중 콜레스테롤.png", bbox_inches='tight')
plt.close()

X = stroke[["HE_chol"]]
Y = stroke[["DI3_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("혈중 콜레스테롤")
plt.ylabel("뇌졸중 진단시기")
plt.grid()
# plt.show()
plt.savefig("res/뇌졸중 진단시기 vs 혈중 콜레스테롤_regression.png", bbox_inches='tight')
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DI3_ag~HE_chol', data=stroke).fit()
print(title)
print(res.summary())

# --------------------------------- stroke / chol

params = ["age", "ainc", "DI3_dg", "DI3_ag", "DI3_pr", "DI3_pt", "HE_chol", "HE_TG"]
stroke = data[params]

mask = (stroke.DI3_ag != 999) & (stroke.DI3_ag != 888) & (stroke.DI3_dg == 1)
stroke = stroke.loc[mask, :]
stroke = stroke[stroke['HE_TG'].notna()]

title = '뇌졸중 진단시기 vs 중성지방'
plt.scatter(stroke["HE_TG"].values, stroke["DI3_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('중성지방')
plt.ylabel('뇌졸중 진단시기')
plt.grid()
# plt.show()
plt.savefig("res/뇌졸중 진단시기 vs 중성지방.png", bbox_inches='tight')
plt.close()

X = stroke[["HE_TG"]]
Y = stroke[["DI3_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("중성지방")
plt.ylabel("뇌졸중 진단시기")
plt.grid()
# plt.show()
plt.savefig("res/뇌졸중 진단시기 vs 중성지방_regression.png", bbox_inches='tight')
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DI3_ag~HE_TG', data=stroke).fit()
print(title)
print(res.summary())

# --------------------------------- lung cancer / smoking

params = ["age", "ainc", "DC6_dg", "DC6_ag", "DC6_pr", "DC6_pt", "BS1_1", "BS2_1", "BS2_2", "BS3_1", "BS3_2",
          "BS3_3", "BS6_2", "BS6_2_1", "BS6_2_2", "BS6_3", "BS6_4", "BS6_4_1", "BS6_4_2", "BD1", "BD2", "BD1_11",
          "BD2_1", "BD2_14", "BD2_31", "BD2_32"]
lungCancer = data[params]

mask = (lungCancer.DC6_ag != 999) & (lungCancer.DC6_ag != 888) & (lungCancer.DC6_dg == 1) & (lungCancer.BS2_2 != 888) & (lungCancer.BS2_2 != 999)
lungCancer = lungCancer.loc[mask, :]

title = '폐암 진단시기 vs 매일흠연 시작연령'
plt.scatter(lungCancer["BS2_2"].values, lungCancer["DC6_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('매일흠연 시작연령')
plt.ylabel('폐암 진단시기')
plt.grid()
# plt.show()
plt.savefig("res/폐암 진단시기 vs 매일흠연 시작연령.png", bbox_inches='tight')
plt.close()

X = lungCancer[["BS2_2"]]
Y = lungCancer[["DC6_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("매일흠연 시작연령")
plt.ylabel("폐암 진단시기")
plt.grid()
# plt.show()
plt.savefig("res/폐암 진단시기 vs 매일흠연 시작연령_regression.png", bbox_inches='tight')
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DC6_ag~BS2_2', data=lungCancer).fit()
print(title)
print(res.summary())

params = ["age", "ainc", "DC2_dg", "DC2_ag", "DC2_pr", "DC2_pt", "BS1_1", "BS2_1", "BS2_2", "BS3_1", "BS3_2",
          "BS3_3", "BS6_2", "BS6_2_1", "BS6_2_2", "BS6_3", "BS6_4", "BS6_4_1", "BS6_4_2", "BD1", "BD2", "BD1_11",
          "BD2_1", "BD2_14", "BD2_31", "BD2_32"]
liverCancer = data[params]
mask = (liverCancer.DC2_ag != 999) & (liverCancer.DC2_ag != 888) & (liverCancer.DC2_dg == 1) & (liverCancer.BD2_31 != 8) & (liverCancer.BD2_31 != 9)
liverCancer = liverCancer.loc[mask, :]

title = '간암 진단시기 vs 폭음 빈도'
plt.scatter(liverCancer["BD2_31"].values, liverCancer["DC2_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('폭음 빈도')
plt.ylabel('간암 진단시기')
plt.grid()
# plt.show()
plt.savefig("res/간암 진단시기 vs 폭음 빈도.png", bbox_inches='tight')
plt.close()

X = liverCancer[["BD2_31"]]
Y = liverCancer[["DC2_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("폭음 빈도")
plt.ylabel("간암 진단시기")
plt.grid()
# plt.show()
plt.savefig("res/간암 진단시기 vs 폭음 빈도_regression.png", bbox_inches='tight')
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DC2_ag~BD2_31', data=liverCancer).fit()
print(title)
print(res.summary())

# --------------------------------- stroke / highBP

params = ["age", "ainc", "DI3_dg", "DI3_ag", "DI3_pr", "DI3_pt", "DI1_dg", "DI1_ag", "DI1_pr", "DI1_pt", "DI1_2"]
stroke = data[params]

mask = (stroke.DI3_ag != 999) & (stroke.DI3_ag != 888) & (stroke.DI3_dg == 1) & (stroke.DI1_ag != 999) & (stroke.DI1_ag != 999)
stroke = stroke.loc[mask, :]
# stroke = stroke[stroke['HE_TG'].notna()]

title = '뇌졸중 진단시기 vs 고혈압 진단시기'
plt.scatter(stroke["DI1_ag"].values, stroke["DI3_ag"].values, c="steelblue", edgecolor="white", s=20)
plt.title(title)
plt.xlabel('고혈압 진단시기')
plt.ylabel('뇌졸중 진단시기')
plt.grid()
# plt.show()
plt.savefig("res/{}.png".format(title), bbox_inches='tight')
plt.close()

X = stroke[["DI1_ag"]]
Y = stroke[["DI3_ag"]]
lr = LinearRegression()
lr.fit(X, Y)
prediction = lr.predict(X)

print("w = ", lr.coef_)
print("b = ", lr.intercept_)
plt.plot(X, lr.predict(X), color='red', lw=2)
plt.scatter(X.values, Y.values, c="steelblue", edgecolor="white", s=30)
plt.title(title)
plt.xlabel("고혈압 진단시기")
plt.ylabel("뇌졸중 진단시기")
plt.grid()
# plt.show()
plt.savefig("res/{}_regression.png".format(title), bbox_inches='tight')
plt.close()

print('Mean_Squared_Error = ', mean_squared_error(prediction, Y))
print('RMSE = ', mean_squared_error(prediction, Y) ** 0.5)

res = ols('DI3_ag~DI1_ag', data=stroke).fit()
print(title)
print(res.summary())
