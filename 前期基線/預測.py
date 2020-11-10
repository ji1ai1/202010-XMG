#python 3.9.0
#python套件 lightgbm 3.0.0
#python套件 numpy 1.19.4
#python套件 pandas 1.1.4
#python套件 scikit-learn 0.23.2
#輸入：
#	data/cust_avli_Q1.csv
#	data/y_Q*_3.csv
#	data/cust_info_q*.csv
#	data/aum_m*.csv
#	data/behavior_m*.csv
#	data/cunkuan_m*.csv
#	data/y_Q*_3.csv
#輸出：
#	result.csv
#
import datetime
import lightgbm
import numpy
import pandas
import random
import sklearn

線上訓練表 = None
for 甲 in [3, 4]:
	甲訓練表 = pandas.read_csv("data/y_Q%d_3.csv" % 甲)
	甲訓練表.columns = ["用戶標識", "標籤"]
	甲訓練表["季度"] = 甲
	線上訓練表 = pandas.concat([線上訓練表, 甲訓練表], ignore_index=True)

線上測試表 = pandas.read_csv("data/cust_avli_Q1.csv", header=0, names=["用戶標識"])
線上測試表["季度"] = 5
線上測試表["標籤"] = -2

用戶表 = None
for 甲 in [3, 4, 5]:
	甲用戶表 = pandas.read_csv("data/cust_info_q%d.csv" % (1 + (甲 - 1) % 4))
	甲用戶表.columns = ["用戶標識", "性別", "年齡", "客戶等級", "本行員工標誌", "職業描述", "我行貸款客戶標誌", "本行產品持有數", "星座描述", "客戶貢獻度", "學歷描述", "家庭年收入", "行業描述", "婚姻狀況描述", "職務描述", "二維碼收單客戶標誌", "VIP客戶標誌", "網銀客戶標誌", "手機銀行客戶標誌", "短信客戶", "微信支付客戶標誌"]
	甲用戶表["季度"] = 甲
	用戶表 = pandas.concat([用戶表, 甲用戶表], ignore_index=True)
類別特征 = ["性別", "客戶等級", "職業描述", "星座描述", "學歷描述", "行業描述", "婚姻狀況描述", "職務描述", ]
for 列名 in 類別特征:
	用戶表[列名] = 用戶表[列名].astype("category")
	
資產表 = None
for 甲 in [3, 4, 5]:
	for 乙 in range(3):
		甲資產表 = pandas.read_csv("data/aum_m%s.csv" % (1 + (3 * 甲 + 乙 - 3) % 12))
		甲資產表["季度"] = 甲
		甲資產表["月度"] = 1 + (3 * 甲 + 乙 - 3)
		資產表 = pandas.concat([資產表, 甲資產表], ignore_index=True)
資產表.columns = ["用戶標識"] + 資產表.columns[1:].to_list()
資產表 = pandas.concat([資產表.loc[:, ["用戶標識", "季度", "月度"]], 資產表.drop(["用戶標識", "季度", "月度"], axis=1)], axis=1)
資產表["X9"] = 資產表.loc[:, ["X%d" % 子 for 子 in range(1, 9)]].sum(axis=1)
資產表["X10"] = (資產表.loc[:, ["X%d" % 子 for 子 in range(1, 9)]] > 0).sum(axis=1)
資產特征序號 = list(range(1, 11))

行為表 = None
for 甲 in [3, 4, 5]:
	for 乙 in range(3):
		甲行為表 = pandas.read_csv("data/behavior_m%s.csv" % (1 + (3 * 甲 + 乙 - 3) % 12))
		甲行為表["季度"] = 甲
		甲行為表["月度"] = 1 + (3 * 甲 + 乙 - 3)
		行為表 = pandas.concat([行為表, 甲行為表], ignore_index=True)
行為表.columns = ["用戶標識"] + 行為表.columns[1:].to_list()
行為表 = pandas.concat([行為表.loc[:, ["用戶標識", "季度", "月度"]], 行為表.drop(["用戶標識", "季度", "月度"], axis=1)], axis=1)
行為表.B6 = [(datetime.datetime.strptime(行為表.B6[子], "%Y-%m-%d %H:%M:%S") - datetime.datetime(year=2019 + 行為表.季度[子] // 4, month=1 + (3 * 行為表.季度[子]) % 12, day=1)).seconds if 行為表.B6[子] is not numpy.nan else numpy.nan for 子 in range(行為表.shape[0])]
行為表["B8"] = 行為表.B2 - 行為表.B4
行為表["B9"] = 行為表.B2 + 行為表.B4
行為表["B10"] = 行為表.B2 / 行為表.B4
行為表["B11"] = 行為表.B3 - 行為表.B5
行為表["B12"] = 行為表.B3 + 行為表.B5
行為表["B13"] = 行為表.B3 / 行為表.B5
行為特征序號 = list(range(1, 14))

存款表 = None
for 甲 in [3, 4, 5]:
	for 乙 in range(3):
		甲存款表 = pandas.read_csv("data/cunkuan_m%s.csv" % (1 + (3 * 甲 + 乙 - 3) % 12))
		甲存款表["季度"] = 甲
		甲存款表["月度"] = 1 + (3 * 甲 + 乙 - 3)
		存款表 = pandas.concat([存款表, 甲存款表], ignore_index=True)
存款表.columns = ["用戶標識"] + 存款表.columns[1:].to_list()
存款表 = pandas.concat([存款表.loc[:, ["用戶標識", "季度", "月度"]], 存款表.drop(["用戶標識", "季度", "月度"], axis=1)], axis=1)
存款特征序號 = list(range(1, 3))




def 取得預備資料表(某表, 某資產表, 某行為表, 某存款表, 標籤月度):
	某表 = 某表.merge(用戶表, on=["用戶標識", "季度"], how="left")
	
	某資料表 = 某表.loc[:, ["用戶標識", "季度", "標籤", "年齡", "本行員工標誌", "我行貸款客戶標誌", "家庭年收入", "二維碼收單客戶標誌", "VIP客戶標誌", "網銀客戶標誌", "手機銀行客戶標誌", "短信客戶", "微信支付客戶標誌"] + 類別特征]
	
	某資產資料表 = 某資產表.loc[某資產表.月度 == 標籤月度]
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 1, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_1" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 2, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_2" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 3, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_3" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 4, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_4" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 5, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_5" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	for 甲 in 資產特征序號:
		某資產資料表["X%d_1_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_1" % 甲]
		某資產資料表["X%d_2_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_2" % 甲]
		某資產資料表["X%d_3_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_3" % 甲]
		某資產資料表["X%d_4_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_4" % 甲]
		某資產資料表["X%d_5_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_5" % 甲]
		某資產資料表["X%d_1_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_1" % 甲]
		某資產資料表["X%d_2_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_2" % 甲]
		某資產資料表["X%d_3_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_3" % 甲]
		某資產資料表["X%d_4_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_4" % 甲]
		某資產資料表["X%d_5_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_5" % 甲]
		某資產資料表 = 某資產資料表.drop(["X%d" % 甲, "X%d_1" % 甲, "X%d_2" % 甲, "X%d_3" % 甲, "X%d_4" % 甲, "X%d_5" % 甲], axis=1)
	某資料表 = 某資料表.merge(某資產資料表, on=["用戶標識", "季度"], how="left")

	某資產季資料表 = 某資產表.groupby(["用戶標識", "季度"]).aggregate({("X%d" % 子): ["sum", "min", "max"] for 子 in 資產特征序號}).reset_index()
	某資產季資料表.columns = ["用戶標識", "季度"] + ["資產季特征%d" % 子 for 子 in range(2, 某資產季資料表.shape[1])]
	某資料表 = 某資料表.merge(某資產季資料表, on=["用戶標識", "季度"], how="left")
	
	某行為季資料表 = 某行為表.groupby(["用戶標識", "季度"]).aggregate({("B%d" % 子): ["sum", "min", "max"] for 子 in 行為特征序號}).reset_index()
	某行為季資料表.columns = ["用戶標識", "季度"] + ["行為季特征%d" % 子 for 子 in range(2, 某行為季資料表.shape[1])]
	某資料表 = 某資料表.merge(某行為季資料表, on=["用戶標識", "季度"], how="left")

	某存款季資料表 = 某存款表.groupby(["用戶標識", "季度"]).aggregate({("C%d" % 子): ["sum", "min", "max", numpy.ptp] for 子 in 存款特征序號}).reset_index()
	某存款季資料表.columns = ["用戶標識", "季度"] + ["存款季特征%d" % 子 for 子 in range(2, 某存款季資料表.shape[1])]
	某資料表 = 某資料表.merge(某存款季資料表, on=["用戶標識", "季度"], how="left")

	return 某資料表

線上預備資料表 = 取得預備資料表(
	線上訓練表.loc[線上訓練表.季度 == 3].copy()
	, 資產表.loc[資產表.季度 == 3].copy()
	, 行為表.loc[行為表.季度 == 3].copy()
	, 存款表.loc[存款表.季度 == 3].copy()
	, 標籤月度 = 9
)
線上預備資料表 = pandas.concat([線上預備資料表.loc[:, ["用戶標識", "季度", "標籤"]], 線上預備資料表.drop(["用戶標識", "季度", "標籤"], axis=1)], axis=1)
線上預備輕模型 = lightgbm.train(train_set=lightgbm.Dataset(線上預備資料表.iloc[:, 4:], label=1 + 線上預備資料表.標籤), num_boost_round=500, params={"objective": "multiclass", "num_classes": 3, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 127, "bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 0, "verbose": -1})

線上訓練預備資料表 = 取得預備資料表(
	線上訓練表.loc[線上訓練表.季度 == 4].copy()
	, 資產表.loc[資產表.季度 == 4].copy()
	, 行為表.loc[行為表.季度 == 4].copy()
	, 存款表.loc[存款表.季度 == 4].copy()
	, 標籤月度 = 12
)
線上訓練預備資料表 = pandas.concat([線上訓練預備資料表.loc[:, ["用戶標識", "季度", "標籤"]], 線上訓練預備資料表.drop(["用戶標識", "季度", "標籤"], axis=1)], axis=1)
線上訓練預備表 = pandas.concat([線上訓練預備資料表.loc[:, ["用戶標識"]], pandas.DataFrame(線上預備輕模型.predict(data=線上訓練預備資料表.iloc[:, 4:]), columns=["預備%d" % 子 for 子 in range(3)])], axis=1)

線上測試預備資料表 = 取得預備資料表(
	線上測試表.loc[線上測試表.季度 == 5].copy()
	, 資產表.loc[資產表.季度 == 5].copy()
	, 行為表.loc[行為表.季度 == 5].copy()
	, 存款表.loc[存款表.季度 == 5].copy()
	, 標籤月度 = 15
)
線上測試預備資料表 = pandas.concat([線上測試預備資料表.loc[:, ["用戶標識", "季度", "標籤"]], 線上測試預備資料表.drop(["用戶標識", "季度", "標籤"], axis=1)], axis=1)
線上測試預備表 = pandas.concat([線上測試預備資料表.loc[:, ["用戶標識"]], pandas.DataFrame(線上預備輕模型.predict(data=線上測試預備資料表.iloc[:, 4:]), columns=["預備%d" % 子 for 子 in range(3)])], axis=1)




def 取得資料表(某表, 某資產表, 某行為表, 某存款表, 某特征表, 某預備表, 標籤月度):
	某表 = 某表.merge(用戶表, on=["用戶標識", "季度"], how="left")
	
	某資料表 = 某表.loc[:, ["用戶標識", "季度", "標籤", "年齡", "本行員工標誌", "我行貸款客戶標誌", "家庭年收入", "二維碼收單客戶標誌", "VIP客戶標誌", "網銀客戶標誌", "手機銀行客戶標誌", "短信客戶", "微信支付客戶標誌"] + 類別特征]
	
	某資產資料表 = 某資產表.loc[某資產表.月度 == 標籤月度]
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 1, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_1" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 2, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_2" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 3, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_3" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 4, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_4" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	某資產資料表 = 某資產資料表.merge(某資產表.loc[某資產表.月度 == 標籤月度 - 5, ["用戶標識"] + ["X%d" % 子 for 子 in 資產特征序號]].rename({("X%d" % 子): ("X%d_5" % 子) for 子 in 資產特征序號}, axis=1), on="用戶標識", how="left")
	for 甲 in 資產特征序號:
		某資產資料表["X%d_1_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_1" % 甲]
		某資產資料表["X%d_2_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_2" % 甲]
		某資產資料表["X%d_3_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_3" % 甲]
		某資產資料表["X%d_4_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_4" % 甲]
		某資產資料表["X%d_5_b" % 甲] = 某資產資料表["X%d" % 甲] / 某資產資料表["X%d_5" % 甲]
		某資產資料表["X%d_1_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_1" % 甲]
		某資產資料表["X%d_2_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_2" % 甲]
		某資產資料表["X%d_3_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_3" % 甲]
		某資產資料表["X%d_4_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_4" % 甲]
		某資產資料表["X%d_5_c" % 甲] = 某資產資料表["X%d" % 甲] - 某資產資料表["X%d_5" % 甲]
		某資產資料表 = 某資產資料表.drop(["X%d" % 甲, "X%d_1" % 甲, "X%d_2" % 甲, "X%d_3" % 甲, "X%d_4" % 甲, "X%d_5" % 甲], axis=1)
	某資料表 = 某資料表.merge(某資產資料表, on=["用戶標識", "季度"], how="left")


	某資產季資料表 = 某資產表.groupby(["用戶標識", "季度"]).aggregate({("X%d" % 子): ["sum", "min", "max"] for 子 in 資產特征序號}).reset_index()
	某資產季資料表.columns = ["用戶標識", "季度"] + ["資產季特征%d" % 子 for 子 in range(2, 某資產季資料表.shape[1])]
	某資料表 = 某資料表.merge(某資產季資料表, on=["用戶標識", "季度"], how="left")
	
	某行為季資料表 = 某行為表.groupby(["用戶標識", "季度"]).aggregate({("B%d" % 子): ["sum", "min", "max"] for 子 in 行為特征序號}).reset_index()
	某行為季資料表.columns = ["用戶標識", "季度"] + ["行為季特征%d" % 子 for 子 in range(2, 某行為季資料表.shape[1])]
	某資料表 = 某資料表.merge(某行為季資料表, on=["用戶標識", "季度"], how="left")

	某存款季資料表 = 某存款表.groupby(["用戶標識", "季度"]).aggregate({("C%d" % 子): ["sum", "min", "max", numpy.ptp] for 子 in 存款特征序號}).reset_index()
	某存款季資料表.columns = ["用戶標識", "季度"] + ["存款季特征%d" % 子 for 子 in range(2, 某存款季資料表.shape[1])]
	某資料表 = 某資料表.merge(某存款季資料表, on=["用戶標識", "季度"], how="left")

	某資料表 = 某資料表.merge(某特征表.loc[:, ["用戶標識", "標籤"]].rename({"標籤": "歷史標籤"}, axis=1), on="用戶標識", how="left")
	某資料表 = 某資料表.merge(某預備表, on="用戶標識", how="left")

	return 某資料表




線上訓練資料表 = 取得資料表(
	線上訓練表.loc[線上訓練表.季度 == 4].copy()
	, 資產表.loc[資產表.季度.isin([3, 4])].copy()
	, 行為表.loc[行為表.季度.isin([3, 4])].copy()
	, 存款表.loc[存款表.季度.isin([3, 4])].copy()
	, 線上訓練表.loc[線上訓練表.季度 == 3].copy()
	, 線上訓練預備表
	, 標籤月度 = 12
)

線上訓練資料表 = pandas.concat([線上訓練資料表.loc[:, ["用戶標識", "季度", "標籤"]], 線上訓練資料表.drop(["用戶標識", "季度", "標籤"], axis=1)], axis=1)
線上第一輕模型 = lightgbm.train(train_set=lightgbm.Dataset(線上訓練資料表.iloc[:, 4:], label=(線上訓練資料表.標籤 == 1).astype("float")), num_boost_round=500, params={"objective": "binary", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 127, "bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 0, "verbose": -1})
線上第二輕模型 = lightgbm.train(train_set=lightgbm.Dataset(線上訓練資料表.loc[線上訓練資料表.標籤 != 1].iloc[:, 4:], label=(線上訓練資料表.loc[線上訓練資料表.標籤 != 1].標籤 == 0).astype("float")), num_boost_round=500, params={"objective": "binary", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 127, "bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 0, "verbose": -1})

線上測試資料表 = 取得資料表(
	線上測試表.copy()
	, 資產表.loc[資產表.季度.isin([4, 5])].copy()
	, 行為表.loc[行為表.季度.isin([4, 5])].copy()
	, 存款表.loc[存款表.季度.isin([4, 5])].copy()
	, 線上訓練表.loc[線上訓練表.季度 == 4].copy()
	, 線上測試預備表
	, 標籤月度 = 15
)
線上測試資料表 = pandas.concat([線上測試資料表.loc[:, ["用戶標識", "季度", "標籤"]], 線上測試資料表.drop(["用戶標識", "季度", "標籤"], axis=1)], axis=1)
線上預測打分表 = 線上測試資料表.loc[:, ["用戶標識"]].copy()
線上預測打分表["第一預測打分"] = 線上第一輕模型.predict(data=線上測試資料表.iloc[:, 4:])
線上預測打分表["第二預測打分"] = 線上第二輕模型.predict(data=線上測試資料表.iloc[:, 4:])

線上預測數量 = [int(0.153 * 線上測試表.shape[0]), int(0.208 * 線上測試表.shape[0]) , int(0.639 * 線上測試表.shape[0])]

線上預測打分表 = 線上預測打分表.sort_values("第一預測打分", ascending=False, ignore_index=True)
線上預測打分表["預測"] = 1
線上預測表 = 線上預測打分表.loc[:(線上預測數量[2] - 1), ["用戶標識", "預測"]]
線上預測打分表 = 線上預測打分表.loc[線上預測數量[2]:].reset_index(drop=True)

線上預測打分表 = 線上預測打分表.sort_values("第二預測打分", ascending=False, ignore_index=True)
線上預測打分表["預測"] = 0
線上預測表 = pandas.concat([線上預測表, 線上預測打分表.loc[:(線上預測數量[1] - 1), ["用戶標識", "預測"]]], ignore_index=True)
線上預測打分表 = 線上預測打分表.loc[線上預測數量[1]:].reset_index(drop=True)

線上預測打分表["預測"] = -1
線上預測表 = pandas.concat([線上預測表, 線上預測打分表.loc[:, ["用戶標識", "預測"]]], ignore_index=True)

提交表 = 線上預測表.loc[:, ["用戶標識", "預測"]]
提交表.columns = ["cust_no", "label"]
提交表.to_csv("result.csv", index=False, header=True)
