### 라이브러리 import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import tensorflow as tf
import matplotlib.pylab as plt # 그래프
from matplotlib import font_manager, rc # 그래프
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# plt 한글깨짐 방지
font_name=font_manager.FontProperties(fname="c:/windows/fonts/malgun.ttf").get_name()
rc('font',family=font_name)





### 프리미어리그 데이터 불러오기
pl = pd.read_csv("https://raw.githubusercontent.com/gyounghwan1313/national-football-team-selection/master/epl_player.csv")
pl.nation=pl.nation.str.split().apply(lambda x: x[1]) # 국적 추출
pl["ateam"]=pl.world.apply(lambda x: 1 if x>0 else 0) # 피파 랭킹 1~20 국가의 대표팀으로 참여하면 1, 아니면 0

'''
17~18 시즌은 17
18~19 시즌은 18
19~20 시즌은 19로 코딩
'''

# 포지션별 인원수 그래프 그리기
plt_table=pl.position.value_counts()
plt.bar(plt_table.index,plt_table,color="black",alpha=0.5)
plt.xlabel("position",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.ylim(0,300)


# 시즌별로 국가대표 선발 여부 crosstab 및 그래프
pl.ateam.value_counts()
pd.crosstab(pl.ateam,pl.season)
stats.chi2_contingency(pd.crosstab(pl.ateam,pl.season)) # 카이스퀘어 검정

group_table=pd.crosstab(pl.season,pl.ateam)
group_table.index=["17~18","18~19","19~10"]

group_table.plot.bar(stacked=True, rot=0)
plt.ylim(150,500)
plt.legend(title="대표팀 선발 여부",labels=["미선발","선발"])
plt.xlabel("EPL 시즌")
plt.ylabel("Count")


# EPL 최대 출장 경기수를 구하고 10경기 미만 출전한 선수는 데이터 셋에서 제외
pl.match.value_counts().index.max()
pl=pl[pl['match']>=10]
pl=pl[pl.position!="GK"] # 골기퍼도 제외

# 변수별로 NA값을 확인 후, 결측값은 0으로 대체
pd.isnull(pl).sum()
pl["ontarget%"].fillna(0,inplace=True)
pl["goal/ontarget"].fillna(0,inplace=True)
pl["goal/shoot"].fillna(0,inplace=True)


# 파생변수 생성
pl["shortpass%"]=pl["shortpass"]/pl["pass"] # 전체 패스 대비 숏패스 비율
pl["mediumpass%"]=pl["mediumpass"]/pl["pass"] # 중거리 패스 비율
pl["longpass%"]=pl["longpass"]/pl["pass"] # 롱패스 비율
pl["keypass%"]=pl["keypass"]/pl["pass"] # 키패스(중요패스) 비율
pl["assist/keypass"]=pl["assist"]/pl["keypass"] # 키패스 수 대비 어시스트 수
pl["assist/cross"]=pl["assist"]/pl["cross"] # 크로스 수 대비 어시스트 수



# 시즌별로 데이터 셋 나누기
pl17=pl[pl.season==17]
pl18=pl[pl.season==18]
pl19=pl[pl.season==19]




### K리그 데이터 불러오기
kl = pd.read_csv("https://raw.githubusercontent.com/gyounghwan1313/national-football-team-selection/master/k_league_player.csv")
kl.match.max() # 최다 출전 경기 수
kl=kl[kl.match>13] # 13번 이하 선수는 삭제
kl=kl[kl.position!="GK"] # 골키퍼는 삭제
kl.position.value_counts() # 포지션별 선수 파악

# 파생 변수 생성
kl["shortpass%"]=kl["shortpass"]/kl["pass"]
kl["mediumpass%"]=kl["mediumpass"]/kl["pass"]
kl["longpass%"]=kl["longpass"]/kl["pass"]
kl["keypass%"]=kl["keypass"]/kl["pass"]
kl["assist/keypass"]=kl["assist"]/kl["keypass"]
kl["assist/cross"]=kl["assist"]/kl["cross"]
kl["time/match"]=kl["time"]/kl["match"]

# 변수별 결측치 확인
pd.isnull(kl).sum()
kl["assist/keypass"].fillna(0,inplace=True)
kl["assist/cross"].fillna(0,inplace=True)
kl.loc[np.isinf(kl['assist/cross']),['assist/cross']]=0 # 무한 값인 경우는 0으로 변경
kl.loc[np.isinf(kl['assist/keypass']),['assist/keypass']]=0 #무한 값인 경우는 0으로 변경


### 공격수 선발

## 공격수 선발 고려 변수
fw_feature= ["goal","ontarget%","shoot/90","ontarget/90","goal/shoot","assist","keypass","keypass%","assist/keypass","assist/cross","offside"]

## 17시즌 공격수
pl17fw=pl17.loc[pl17["position"]=="FW",["season","name","position"]+fw_feature+["ateam"]]
pd.isnull(pl17fw).sum()
pl17fw.loc[np.isinf(pl17fw['assist/cross']),['assist/cross']] =0 # 무한은 0으로 변환
pl17fw.loc[pd.isnull(pl17fw['assist/cross']),['assist/cross']] =0 # 결측치는 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,14)):
    pl17fw.iloc[:,i] = pl17fw.iloc[:,i]/pl17fw.iloc[:,i].mean()

##18시즌 공격수
pl18fw=pl18.loc[pl18["position"]=="FW",["season","name","position"]+fw_feature+["ateam"]]
pl18fw.loc[np.isinf(pl18fw['assist/cross']),['assist/cross']] =0 # 무한은 0으로 변환
pl18fw.loc[pd.isnull(pl18fw['assist/cross']),['assist/cross']] =0 # 결측치는 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,14)):
    pl18fw.iloc[:,i] = pl18fw.iloc[:,i]/pl18fw.iloc[:,i].mean()


## 19시즌 공격수
pl19fw=pl19.loc[pl19["position"]=="FW",["season","name","position"]+fw_feature+["ateam"]]
pl19fw.loc[np.isinf(pl19fw['assist/cross']),['assist/cross']] =0 # 무한은 0으로 변환
pl19fw.loc[pd.isnull(pl19fw['assist/cross']),['assist/cross']] =0 # 결측치는 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,14)):
    pl19fw.iloc[:,i] = pl19fw.iloc[:,i]/pl19fw.iloc[:,i].mean()

##공격수 전체 시즌
plfw=pd.concat([pl17fw,pl18fw,pl19fw])
plfw=plfw[plfw.goal!=0] # 골이 0인 공격수는 제외


## EPL 데이터 스플릿
fw_x_train,fw_x_test,fw_y_train,fw_y_test=train_test_split(plfw.iloc[:,3:-1],plfw["ateam"],test_size=0.2,random_state=11,stratify=plfw["ateam"])


## 한국 공격수 데이터도 EPL데이터와 동일하게 편집
klfw=kl.loc[kl["position"]=="FW",["season","name","position"]+fw_feature]

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,14)):
    klfw.iloc[:,i] = klfw.iloc[:,i]/klfw.iloc[:,i].mean()



## Desicion tree로 선발 선수 예측
decision_model=DecisionTreeClassifier(criterion='entropy',max_depth=5)
decision_model.fit(fw_x_train,fw_y_train)

decision_model.score(fw_x_train,fw_y_train) #0.9148936170212766
decision_model.score(fw_x_test,fw_y_test) # 0.7708333333333334

kl_decision_predict=decision_model.predict(klfw[fw_feature])

klfw["ateam"] = list(kl_decision_predict)
klfw.loc[(klfw["ateam"]==1) & klfw["goal"]>=1,'name'] # 선발 후보 명단
'''
26      세징야_대구
90     일류첸코_포항
126     펠리페_광주
162     무고사_인천
180     주니오_울산
'''


## 랜덤포레스트로 선발 선수 예측
rf_model=RandomForestClassifier(criterion="entropy",max_depth=7,n_estimators=100,oob_score=True)
rf_model.fit(fw_x_train,fw_y_train)

rf_model.score(fw_x_train,fw_y_train) #0.9787234042553191
rf_model.score(fw_x_test,fw_y_test) # 0.8125

kl_rf_predict=rf_model.predict(klfw[fw_feature])

klfw["ateam"] = list(kl_rf_predict)
klfw.loc[(klfw["ateam"]==1) & klfw["goal"]>0,'name'] # 선발 후보 명단
'''
26      세징야_대구
90     일류첸코_포항
162     무고사_인천
180     주니오_울산
'''




## CNN으로 선발 선수 예측
ar_fw_x_train=np.array(fw_x_train)
ar_fw_y_train=np.array(pd.get_dummies(fw_y_train))

model = Sequential()
model.add(Dense(128,input_dim=11))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
model.fit(ar_fw_x_train,ar_fw_y_train,epochs=70,shuffle=False,batch_size=32)

pd.crosstab(fw_y_test,np.argmax(model.predict(fw_x_test),axis=1))
print(classification_report(fw_y_test,np.argmax(model.predict(fw_x_test),axis=1))) #0.76

klfw["ateam"] = list(np.argmax(model.predict(np.array(klfw[fw_feature])),axis=1))
klfw.loc[klfw["ateam"]==1,'name'] # 선발 후보 명단
'''
26     세징야_대구
89     고무열_강원
126    펠리페_광주
146    나상호_성남
162    무고사_인천
180    주니오_울산
'''

# Logistic regression으로 공격수 선발 예측
scaler = StandardScaler() #인스턴스
fw_x_train_std = scaler.fit_transform(fw_x_train) # z-score로 변환
fw_x_test_std = scaler.transform(fw_x_test) # z-score로 변환

lg = LogisticRegression(solver='lbfgs',max_iter=3,penalty='l2')
lg.fit(fw_x_train_std, fw_y_train)
lg.score(fw_x_train_std, fw_y_train) #0.8658536585365854
lg.score(fw_x_test,fw_y_test) #0.5714285714285714

kl_lg_predict=lg.predict(klfw[fw_feature])
klfw["ateam"] = list(kl_lg_predict)
klfw.loc[klfw["ateam"]==1,'name'] # 선발 후보 명단
'''
26      세징야_대구
89      고무열_강원
104     김승대_강원
126     펠리페_광주
146     나상호_성남
159      데얀_대구
162     무고사_인천
180     주니오_울산
182     이정협_부산
253    비욘존슨_울산
'''


# SVM으로 공격수 선발 예측
svm_clf =svm.SVC(kernel = 'rbf')
svm_clf.fit(fw_x_train_std, fw_y_train)
svm_clf.score(fw_x_train_std, fw_y_train) # 0.9085365853658537
svm_clf.score(fw_x_test,fw_y_test) # 0.6904761904761905

kl_svm_predict=svm_clf.predict(klfw[fw_feature])
klfw["ateam"] = list(kl_svm_predict)
klfw.loc[(klfw["ateam"]==1) & klfw["goal"]>0,'name'] # 선발 후보 명단
'''
26     세징야_대구
89     고무열_강원
126    펠리페_광주
146    나상호_성남
162    무고사_인천
180    주니오_울산
'''




###  미드필더 선발

# 미드필더 선발 고려 변수
mf_feature=["goal","pass","mediumpass%","longpass%","assist", "keypass", "cross","assist/cross","assist/keypass","intercept"]

# 17시즌 미드필더
pl17mf = pl17.loc[pl17["position"]=="MF",["season","name","position","goal","pass","mediumpass%","longpass%","assist", "keypass", "cross","assist/cross","assist/keypass","intercept","ateam"]]
pl17mf.loc[np.isinf(pl17mf['assist/cross']),['assist/cross']]=0 # 무한값 0으로 변환
pl17mf.loc[pd.isnull(pl17mf['assist/cross']),['assist/cross']]=0 # 결측치 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,13)):
    pl17mf.iloc[:,i] = pl17mf.iloc[:,i]/pl17mf.iloc[:,i].mean()

# 18시즌 미드필더
pl18mf = pl18.loc[pl18["position"]=="MF",["season","name","position","goal","pass","mediumpass%","longpass%","assist", "keypass", "cross","assist/cross","assist/keypass","intercept","ateam"]]
pl18mf.loc[np.isinf(pl18mf['assist/cross']),['assist/cross']]=0 # 무한값 0으로 변환
pl18mf.loc[pd.isnull(pl18mf['assist/cross']),['assist/cross']]=0 # 결측치 0으로 변환
pl18mf.loc[pd.isnull(pl18mf['assist/keypass']),['assist/keypass']]=0 # 결측치 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,13)):
    pl18mf.iloc[:,i] = pl18mf.iloc[:,i]/pl18mf.iloc[:,i].mean()

# 19시즌 미드필더
pl19mf = pl19.loc[pl19["position"]=="MF",["season","name","position","goal","pass","mediumpass%","longpass%","assist", "keypass", "cross","assist/cross","assist/keypass","intercept","ateam"]]
pl19mf.loc[np.isinf(pl19mf['assist/cross']),['assist/cross']]=0 # 무한값 0으로 변환
pl19mf.loc[pd.isnull(pl19mf['assist/cross']),['assist/cross']]=0 # 결측치 0으로 변환
pl19mf.loc[pd.isnull(pl19mf['assist/keypass']),['assist/keypass']]=0 # 결측치 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,13)):
    pl19mf.iloc[:,i] = pl19mf.iloc[:,i]/pl19mf.iloc[:,i].mean()


## 미드필더 전체 시즌
plmf=pd.concat([pl17mf,pl18mf,pl19mf])

## k리그도 동일한 변수를 불러오고 편집
klmf=kl.loc[kl["position"]=="MF",["season","name","position","goal","pass","mediumpass%","longpass%","assist", "keypass", "cross","assist/cross","assist/keypass","intercept"]]
for i in list(range(3,13)):
    klmf.iloc[:,i] = klmf.iloc[:,i]/klmf.iloc[:,i].mean()


# Epl 선수 데이터  split
mf_x_train,mf_x_test,mf_y_train,mf_y_test=train_test_split(plmf[mf_feature],plmf["ateam"],test_size=0.2,random_state=11,stratify=plmf["ateam"])


# decision tree로 미드필더 선발 예측
decision_model=DecisionTreeClassifier(criterion='entropy',max_depth=7)
decision_model.fit(mf_x_train,mf_y_train)

decision_model.score(mf_x_train,mf_y_train) #0.9112149532710281
decision_model.score(mf_x_test,mf_y_test) # 0.7407407407407407

kl_decision_predict=decision_model.predict(klmf[mf_feature])

klmf["ateam"] = list(kl_decision_predict)
klmf.loc[klmf["ateam"]==1,'name'] # 미드필더 선발 후보 명단
'''
1        손준호_전북
3        한국영_강원
35       김보경_전북
37       김동현_성남
41     팔로세비치_포항
131      고명진_울산
'''


# 랜덤포레스트로 미드필더 선발 예측
rf_model=RandomForestClassifier(criterion="entropy",max_depth=7,n_estimators=100,oob_score=True)
rf_model.fit(mf_x_train,mf_y_train)

rf_model.score(mf_x_train,mf_y_train) #0.9766355140186916
rf_model.score(mf_x_test,mf_y_test) # 0.8333333333333334

kl_rf_predict=rf_model.predict(klmf[mf_feature])
klmf["ateam"] = list(kl_rf_predict)
klmf.loc[klmf["ateam"]==1,'name'] # 미드필더 선발 후보 명단
'''
1     손준호_전북
3     한국영_강원
7     최영준_포항
27    고승범_수원
'''



# CNN로 미드필더 선발 예측
ar_mf_x_train=np.array(mf_x_train)
ar_mf_y_train=np.array(pd.get_dummies(mf_y_train))

model = Sequential()
model.add(Dense(128,input_dim=10))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
model.fit(ar_mf_x_train,ar_mf_y_train,epochs=50,shuffle=False,batch_size=32)

pd.crosstab(mf_y_test,np.argmax(model.predict(mf_x_test),axis=1))
print(classification_report(mf_y_test,np.argmax(model.predict(mf_x_test),axis=1))) #0.85

klmf["ateam"] = list(np.argmax(model.predict(np.array(klmf[mf_feature])),axis=1))
klmf.loc[klmf["ateam"]==1,'name'] # 미드필더 선발 후보 명단
'''
1         손준호_전북
3         한국영_강원
17        신진호_울산
27        고승범_수원
37        김동현_성남
39        김도혁_인천
41      팔로세비치_포항
55     이스칸데로프_성남
85        박상혁_수원
131       고명진_울산
169       임민혁_광주
'''


# Logistic regression으로 미드필더 선발 예측
mf_x_train_std = scaler.fit_transform(mf_x_train) # z-score
mf_x_test_std = scaler.transform(mf_x_test) # z-score

lg = LogisticRegression(solver='lbfgs',max_iter=4,C=1,penalty='l2')
lg.fit(mf_x_train_std, mf_y_train)
lg.score(mf_x_train_std, mf_y_train) # 0.7710280373831776
lg.score(mf_x_test_std, mf_y_test) # 0.8518518518518519

kl_lg_predict=lg.predict(klmf[mf_feature])
klmf["ateam"] = list(kl_lg_predict)
klmf.loc[klmf["ateam"]==1,'name'] # 미드필더 선발 후보 명단
'''
1        손준호_전북
17       신진호_울산
20       호물로_부산
27       고승범_수원
35       김보경_전북
39       김도혁_인천
41     팔로세비치_포항
73       이재권_강원
84       이승모_포항
131      고명진_울산
'''

# SVM으로 미드필더 선발 예측

svm_clf =svm.SVC(kernel = 'rbf')
svm_clf.fit(mf_x_train_std, mf_y_train)
svm_clf.score(mf_x_train_std, mf_y_train) #.8271028037383178
svm_clf.score(mf_x_test,mf_y_test) #0.7777777777777778

kl_svm_predict=svm_clf.predict(klmf[mf_feature])
klmf["ateam"] = list(kl_svm_predict)
klmf.loc[klmf["ateam"]==1,'name'] #  미드필더 선발 후보 명단
'''
1         손준호_전북
17        신진호_울산
37        김동현_성남
55     이스칸데로프_성남
67        박종우_부산
68        주세종_서울
76       아길라르_인천
169       임민혁_광주
'''



##  윙어 선발

# 윙어 선발 변수
w_feature=["goal", "ontarget%", "shoot/90", "ontarget/90", "goal/shoot", "goal/ontarget", "assist","keypass", "gainfoul","offside", "keypass%", "assist/keypass", "assist/cross"]


# 17시즌 윙어
pl17w = pl17.loc[pl17["position"]=="W",["season","name","position","goal", "ontarget%", "shoot/90", "ontarget/90", "goal/shoot", "goal/ontarget", "assist","keypass", "gainfoul","offside", "keypass%", "assist/keypass", "assist/cross","ateam"]]
pl17w.loc[np.isinf(pl17w['assist/cross']),['assist/cross']] =0 # 무한값 0으로 변환
pl17w.loc[pd.isnull(pl17w['assist/cross']),['assist/cross']] =0 #결측치 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,16)):
    pl17w.iloc[:,i] = pl17w.iloc[:,i]/pl17w.iloc[:,i].mean()

# 18시즌 윙어
pl18w = pl18.loc[pl18["position"]=="W",["season","name","position","goal", "ontarget%", "shoot/90", "ontarget/90", "goal/shoot", "goal/ontarget", "assist","keypass", "gainfoul","offside", "keypass%", "assist/keypass", "assist/cross","ateam"]]
pl18w.loc[np.isinf(pl18w['assist/cross']),['assist/cross']] =0 # 무한값 0으로 변환
pl18w.loc[pd.isnull(pl18w['assist/cross']),['assist/cross']] =0 # 결측치 0으로 변환
pl18w.loc[pd.isnull(pl18w['assist/keypass']),['assist/keypass']] =0 # 결측치 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,16)):
    pl18w.iloc[:,i] = pl18w.iloc[:,i]/pl18w.iloc[:,i].mean()

# 19시즌 윙어
pl19w = pl19.loc[pl19["position"]=="W",["season","name","position","goal", "ontarget%", "shoot/90", "ontarget/90", "goal/shoot", "goal/ontarget", "assist","keypass", "gainfoul","offside", "keypass%", "assist/keypass", "assist/cross","ateam"]]
pl19w.loc[np.isinf(pl19w['assist/cross']),['assist/cross']] =0 # 무한값 0으로 변환
pl19w.loc[pd.isnull(pl19w['assist/cross']),['assist/cross']] =0 # 결측치 0으로 변환
pl19w.loc[pd.isnull(pl19w['assist/keypass']),['assist/keypass']] =0 # 결측치 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,16)):
    pl19w.iloc[:,i] = pl19w.iloc[:,i]/pl19w.iloc[:,i].mean()

# 전체 시즌 윙어
plw=pd.concat([pl17w,pl18w,pl19w])


## K리그 선수들도 동일한 변수를 불러옴
klw=kl.loc[kl["position"]=="W",["season","name","position","goal", "ontarget%", "shoot/90", "ontarget/90", "goal/shoot", "goal/ontarget", "assist","keypass", "gainfoul","offside", "keypass%", "assist/keypass", "assist/cross"]]

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,16)):
    klw.iloc[:,i] = klw.iloc[:,i]/klw.iloc[:,i].mean()


## EPL 윙어 데이터 스플릿
w_x_train,w_x_test,w_y_train,w_y_test=train_test_split(plw[w_feature],plw["ateam"],test_size=0.2,random_state=11,stratify=plw["ateam"])


# Decision tree로 윙어 선발 예측
decision_model=DecisionTreeClassifier(criterion='entropy',max_depth=7)
decision_model.fit(w_x_train,w_y_train)

decision_model.score(w_x_train,w_y_train) # 0.957983193277311
decision_model.score(w_x_test,w_y_test) # 0.6666666666666666

kl_decision_predict=decision_model.predict(klw[w_feature])
klw["ateam"] = list(kl_decision_predict)
klw.loc[klw["ateam"]==1,'name'] # 윙어 선발 후보 명단
'''
50     염기훈_수원
139    이동준_부산
198    송시우_인천
208    김호남_인천
'''


# 랜덤포레스트로 윙어 선발 예측
rf_model=RandomForestClassifier(criterion="entropy",max_depth=5,n_estimators=100,oob_score=True,random_state=100)
rf_model.fit(w_x_train,w_y_train)
rf_model.score(w_x_train,w_y_train) # 0.9831932773109243
rf_model.score(w_x_test,w_y_test) # 0.7333333333333333

kl_rm_predict=rf_model.predict(klw[w_feature])
klw["ateam"] = list(kl_rm_predict)
klw.loc[klw["ateam"]==1,'name'] # 윙어 선발 후보 명단
'''
50     염기훈_수원
139    이동준_부산
'''

# CNN으로 윙어 선발 예측
ar_w_x_train=np.array(w_x_train)
ar_w_y_train=np.array(pd.get_dummies(w_y_train))

model = Sequential()
model.add(Dense(128,input_dim=13))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
model.fit(ar_w_x_train,ar_w_y_train,epochs=70,shuffle=False,batch_size=32)

pd.crosstab(w_y_test,np.argmax(model.predict(w_x_test),axis=1))
print(classification_report(w_y_test,np.argmax(model.predict(w_x_test),axis=1))) # 0.63
klw["ateam"] = list(np.argmax(model.predict(np.array(klw[w_feature])),axis=1))
klw.loc[klw["ateam"]==1,'name'] # 윙어 선발 후보 명단
'''
33     이승기_전북
77     송민규_포항
101    한교원_전북
'''



# Logistic regression으로 미드필더 선발 예측
w_x_train_std = scaler.fit_transform(w_x_train)
w_x_test_std = scaler.transform(w_x_test)

lg = LogisticRegression(solver='lbfgs',max_iter=4,C=1,penalty='l2')
lg.fit(w_x_train_std, w_y_train)
lg.score(w_x_train_std, w_y_train) # 0.8319327731092437
lg.score(w_x_test_std, w_y_test) # 0.7666666666666667

kl_lg_predict=lg.predict(klw[w_feature])
klw["ateam"] = list(kl_lg_predict)
klw.loc[klw["ateam"]==1,'name'] # 윙어 선발 후보 명단
'''
50     염기훈_수원
77     송민규_포항
101    한교원_전북
139    이동준_부산
'''

# SVM으로 윙어 선발 예측
svm_clf = svm.SVC(kernel = 'rbf',random_state=5)
parameters = {'C': [ 0.01, 0.1, 1, 10, 50, 100],
             'gamma':[ 0.01, 0.1, 1, 10, 50, 100]}

grid_svm = GridSearchCV(svm_clf,
                      param_grid = parameters, cv = 5)

grid_svm.fit(w_x_train_std, w_y_train)
grid_svm.score(w_x_train_std, w_y_train) # 0.8235294117647058
grid_svm.score(w_x_test,w_y_test) # 0.7333333333333333

kl_svm_predict=grid_svm.predict(klw[w_feature])
klw["ateam"] = list(kl_svm_predict)
klw.loc[klw["ateam"]==1,'name'] # 윙어 선발 후보 명단
'''
139    이동준_부산
'''



## 중앙수비수 선발

# 중앙 수비수 선발 변수
cb_feature=["tackle","cut","block","intercept","clearing","foul"]


# 17시즌 중앙수비수
pl17cb = pl17.loc[pl17["position"]=="DF",["season","name","position","tackle","cut","block","intercept","clearing","foul","ateam"]]

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,9)):
    pl17cb.iloc[:,i] = pl17cb.iloc[:,i]/pl17cb.iloc[:,i].mean()

# 18시즌 중앙수비수
pl18cb = pl18.loc[pl18["position"]=="DF",["season","name","position","tackle","cut","block","intercept","clearing","foul","ateam"]]

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,9)):
    pl18cb.iloc[:,i] = pl18cb.iloc[:,i]/pl18cb.iloc[:,i].mean()

# 19시즌 중앙수비수
pl19cb = pl19.loc[pl19["position"]=="DF",["season","name","position","tackle","cut","block","intercept","clearing","foul","ateam"]]

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,9)):
    pl19cb.iloc[:,i] = pl19cb.iloc[:,i]/pl19cb.iloc[:,i].mean()

## EPL 정체 중앙 수비수
plcb=pd.concat([pl17cb,pl18cb,pl19cb])

## K리그 중앙 수비수도 동일한 변수로 불러옴
klcb=kl.loc[kl["position"]=="DF",["season","name","position","tackle","cut","block","intercept","clearing","foul","keypass"]]

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,9)):
    klcb.iloc[:,i] = klcb.iloc[:,i]/klcb.iloc[:,i].mean()


## EPL 중앙 수비스 데이터 스플릿
cb_x_train,cb_x_test,cb_y_train,cb_y_test=train_test_split(plcb[cb_feature],plcb["ateam"],test_size=0.2,random_state=11,stratify=plcb["ateam"])


# Decision tree로 중앙 수비수 선발 예측
decision_model=DecisionTreeClassifier(criterion='entropy',max_depth=7)
decision_model.fit(cb_x_train,cb_y_train)

decision_model.score(cb_x_train,cb_y_train) #  0.8899521531100478
decision_model.score(cb_x_test,cb_y_test) # 0.6226415094339622

kl_decision_predict=decision_model.predict(klcb[cb_feature])
klcb["ateam"] = list(kl_decision_predict)
klcb.loc[(klcb["ateam"]==1),'name'] # 중앙 수비수 선발 후보 명단
'''
0     김영빈_강원
13    권경원_상주
15    정승현_울산
28    홍정호_전북
29    김우석_대구
57     헨리_수원
72    임승겸_성남
'''


# 랜덤포레스트로 중앙 수비수 선발 예측
rf_model=RandomForestClassifier(criterion="entropy",max_depth=8,n_estimators=100,oob_score=True,random_state=11)
rf_model.fit(cb_x_train,cb_y_train)

rf_model.score(cb_x_train,cb_y_train) #0.9473684210526315
rf_model.score(cb_x_test,cb_y_test) # 0.6792452830188679

kl_rm_predict=rf_model.predict(klcb[cb_feature])
klcb["ateam"] = list(kl_rm_predict)
klcb.loc[klcb["ateam"]==1,'name']# 중앙 수비수 선발 후보 명단
'''
36    김남춘_서울
57     헨리_수원
'''




# CNN으로 중앙 수비수 선발 예측
ar_cb_x_train=np.array(cb_x_train)
ar_cb_y_train=np.array(pd.get_dummies(cb_y_train))

model = Sequential()
model.add(Dense(128,input_dim=6))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
model.fit(ar_cb_x_train,ar_cb_y_train,epochs=100,shuffle=False,batch_size=32)

pd.crosstab(cb_y_test,np.argmax(model.predict(cb_x_test),axis=1))
print(classification_report(cb_y_test,np.argmax(model.predict(cb_x_test),axis=1))) #0.62

klcb["ateam"] = list(np.argmax(model.predict(np.array(klcb[cb_feature])),axis=1))
klcb.loc[klcb["ateam"]==1,'name'] # 중앙 수비수 선발 후보 명단
'''
5     연제운_성남
28    홍정호_전북
'''


# Logistic Regression 로 중앙 수비수 선발 예측
cb_x_train_std = scaler.fit_transform(cb_x_train) #z-score
cb_x_test_std = scaler.transform(cb_x_test) #z-score

lg = LogisticRegression(solver='lbfgs',max_iter=3,C=1.2,penalty='l2')
lg.fit(cb_x_train_std, cb_y_train)
lg.score(cb_x_train_std, cb_y_train) #0.7081339712918661
lg.score(cb_x_test_std, cb_y_test) #0.7169811320754716

kl_lg_predict=lg.predict(klcb[cb_feature])
klcb["ateam"] = list(kl_lg_predict)
klcb.loc[klcb["ateam"]==1,'name'] # 중앙 수비수 선발 후보 명단 - 후보 없음


#SVM로 중앙 수비수 선발 예측
svm_clf = svm.SVC(kernel = 'rbf',random_state=5)
parameters = {'C': [ 0.01, 0.1, 1, 5, 10, 50],
             'gamma':[ 0.01, 0.1, 1, 5, 10, 50]}

grid_svm = GridSearchCV(svm_clf,
                      param_grid = parameters, cv = 5)

grid_svm.fit(cb_x_train_std, cb_y_train)
grid_svm.score(cb_x_train_std, cb_y_train) # 0.9856459330143541
grid_svm.score(cb_x_test,cb_y_test) #0.7169811320754716

kl_svm_predict=grid_svm.predict(klcb[cb_feature])
klcb["ateam"] = list(kl_svm_predict)
klcb.loc[klcb["ateam"]==1,'name'] # 중앙 수비수 선발 후보 명단 - 후보 없음




##  측면수비수 선발

# 측면수비수 선발 변수
wb_feature=["assist","keypass","cross","cut","block","intercept","clearing","tackle","foul","assist/cross","assist/keypass"]

# 17시즌 측면 수비수
pl17wb = pl17.loc[pl17["position"]=="FB",["season","name","position","assist","keypass","cross","cut","block","intercept","clearing","tackle","foul","assist/cross","assist/keypass","keypass%","ateam"]]
pl17wb.loc[np.isinf(pl17wb['assist/cross']),['assist/cross']] =0 # 무한값 0으로 변환
pl17wb.loc[pd.isnull(pl17wb['assist/cross']),['assist/cross']] =0 # 결측기 0으로 변환
pl17wb.loc[pd.isnull(pl17wb['assist/keypass']),['assist/keypass']] =0 # 결측기 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,15)):
    pl17wb.iloc[:,i] = pl17wb.iloc[:,i]/pl17wb.iloc[:,i].mean()

#18시즌 측면 수비수
pl18wb = pl18.loc[pl18["position"]=="FB",["season","name","position","assist","keypass","cross","cut","block","intercept","clearing","tackle","foul","assist/cross","assist/keypass","keypass%","ateam"]]
pl18wb.loc[np.isinf(pl18wb['assist/cross']),['assist/cross']] =0 # 무한값 0으로 변환
pl18wb.loc[pd.isnull(pl18wb['assist/cross']),['assist/cross']] =0 # 결측기 0으로 변환
pl18wb.loc[pd.isnull(pl18wb['assist/keypass']),['assist/keypass']] =0 # 결측기 0으로 변환
pl18wb.loc[pd.isnull(pl18wb['assist/keypass']),['assist/keypass']] =0 # 결측기 0으로 변환

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,15)):
    pl18wb.iloc[:,i] = pl18wb.iloc[:,i]/pl18wb.iloc[:,i].mean()

#19시즌 측면 수비수
pl19wb = pl19.loc[pl19["position"]=="FB",["season","name","position","assist","keypass","cross","cut","block","intercept","clearing","tackle","foul","assist/cross","assist/keypass","keypass%","ateam"]]
pl19wb.loc[np.isinf(pl19wb['assist/cross']),['assist/cross']] =0 # 무한값 0으로 변환
pl19wb.loc[pd.isnull(pl19wb['assist/cross']),['assist/cross']] =0 # 결측기 0으로 변환
pl19wb.loc[pd.isnull(pl19wb['assist/keypass']),['assist/keypass']] =0 # 결측기 0으로 변환


# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,15)):
    pl19wb.iloc[:,i] = pl19wb.iloc[:,i]/pl19wb.iloc[:,i].mean()

##  전체 측면 수비수
plwb=pd.concat([pl17wb,pl18wb,pl19wb])


## K리그 측면 수비수도 동일한 방식으로 편집
klwb=kl.loc[kl["position"]=="FB",["season","name","position","assist","keypass","cross","cut","block","intercept","clearing","tackle","foul","assist/cross","assist/keypass","keypass%"]]

# 각 선수, 컬럼별로 해당 시즌의 평균값을 나눔, 해당 시즌에 리그 평균별 자신의 점수를 계산
for i in list(range(3,15)):
    klwb.iloc[:,i] = klwb.iloc[:,i]/klwb.iloc[:,i].mean()


## EPL 선수 데이터 스플릿
wb_x_train,wb_x_test,wb_y_train,wb_y_test=train_test_split(plwb[wb_feature],plwb["ateam"],test_size=0.2,random_state=11,stratify=plwb["ateam"])


# Decision tree로 측면 수비수 선발 예측
decision_model=DecisionTreeClassifier(criterion='entropy',max_depth=7)
decision_model.fit(wb_x_train,wb_y_train)

decision_model.score(wb_x_train,wb_y_train) # 0.9391891891891891
decision_model.score(wb_x_test,wb_y_test) # 0.8421052631578947

kl_decision_predict=decision_model.predict(klwb[wb_feature])
klwb["ateam"] = list(kl_decision_predict)
klwb.loc[(klwb["ateam"]==1),'name'] # 측면 수비수 선발 후보
'''
12     이용_전북
25    강상우_포항
38    정승원_대구
48    김진수_전북
74    양상민_수원
'''


#랜덤포레스트로 측면 수비수 선발 예측
rf_model=RandomForestClassifier(criterion="entropy",max_depth=7,n_estimators=100,oob_score=True)
rf_model.fit(wb_x_train,wb_y_train)

rf_model.score(wb_x_train,wb_y_train) # 0.9662162162162162
rf_model.score(wb_x_test,wb_y_test) # 0.7631578947368421

kl_rm_predict=rf_model.predict(klwb[wb_feature])
klwb["ateam"] = list(kl_rm_predict)
klwb.loc[klwb["ateam"]==1,'name'] # 측면 수비수 선발 후보
'''
25    강상우_포항
38    정승원_대구
'''

# CNN으로 측면 수비수 선발 예측
ar_wb_x_train=np.array(wb_x_train)
ar_wb_y_train=np.array(pd.get_dummies(wb_y_train))

model = Sequential()
model.add(Dense(128,input_dim=11)) #dim에는 열의 개수
model.add(Dense(128,activation='relu')) #dim에는 열의 개수
model.add(Dense(128,activation='relu')) #dim에는 열의 개수
model.add(Dense(64,activation='relu')) #dim에는 열의 개수
model.add(Dense(64,activation='relu')) #dim에는 열의 개수
model.add(Dense(64,activation='relu')) #dim에는 열의 개수
model.add(Dense(4,activation='relu')) #dim에는 열의 개수
model.add(Dense(2,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
model.fit(ar_wb_x_train,ar_wb_y_train,epochs=80,shuffle=False,batch_size=32)

pd.crosstab(wb_y_test,np.argmax(model.predict(wb_x_test),axis=1))
print(classification_report(wb_y_test,np.argmax(model.predict(wb_x_test),axis=1))) #0.84

model.predict(np.array(klwb[wb_feature]))
klwb["ateam"] = list(np.argmax(model.predict(np.array(klwb[wb_feature])),axis=1))
klwb.loc[klwb["ateam"]==1,'name'] # 측면 수비수 선발 후보
'''
12      이용_전북
16     김태환_울산
25     강상우_포항
38     정승원_대구
48     김진수_전북
149    이으뜸_광주
152    이상준_부산
'''




# Logistic regression으로 측면 수비수 선발 예측
wb_x_train_std = scaler.fit_transform(wb_x_train) #z-score
wb_x_test_std = scaler.transform(wb_x_test) #z-score

lg = LogisticRegression(solver='lbfgs',max_iter=4,C=1,penalty='l2')
lg.fit(wb_x_train_std, wb_y_train)
lg.score(wb_x_train_std, wb_y_train) # 0.8108108108108109
lg.score(wb_x_test_std, wb_y_test) # 0.7631578947368421

kl_lg_predict=lg.predict(klwb[wb_feature])
klwb["ateam"] = list(kl_lg_predict)
klwb.loc[klwb["ateam"]==1,'name'] # 측면 수비수 선발 후보
'''
16     김태환_울산
25     강상우_포항
38     정승원_대구
66      홍철_울산
149    이으뜸_광주
'''


# SVM로 측면 수비수 선발 예측
svm_clf = svm.SVC(kernel = 'rbf',random_state=5)
parameters = {'C': [ 0.01, 0.1, 1, 10, 50, 100],
             'gamma':[ 0.01, 0.1, 1, 10, 50, 100]}

grid_svm = GridSearchCV(svm_clf,
                      param_grid = parameters, cv = 5)

grid_svm.fit(wb_x_train_std, wb_y_train)
grid_svm.score(wb_x_train_std, wb_y_train) # 0.831081081081081
grid_svm.score(wb_x_test,wb_y_test) # 0.7631578947368421

kl_svm_predict=grid_svm.predict(klwb[wb_feature])
klwb["ateam"] = list(kl_svm_predict)
klwb.loc[klwb["ateam"]==1,'name'] # 측면 수비수 선발 후보
'''
25    강상우_포항
38    정승원_대구
'''




### 2018년 러시아 월드컵 포메이션 정보로 포지션 선택
data = pd.read_csv("https://raw.githubusercontent.com/gyounghwan1313/national-football-team-selection/master/2018_russia_worldcup.csv")
data_win=data[data.구분=="승"]
data_draw=data[data.구분=="무"]


# 가장 승리가 많은 포메이션
data_win['승리팀_포메이션'].value_counts()

# 패배가 가장 많은 포메이션
data_win['패배팀_포메이션'].value_counts()


# 무승무가 가장 많은 포메이션
data_draw["승리팀_포메이션"].value_counts().add(data_draw["패배팀_포메이션"].value_counts(),fill_value=0)


#조별예선 단계에서 승리가 가장 많은 포메이션
data_win.loc[data_win["단계"]==32,'승리팀_포메이션'].value_counts()

#조별예선 단계에서 패배가 가장 많은 포메이션
data_win.loc[data_win["단계"]==32,'패배팀_포메이션'].value_counts()

#32강 승률
x=data_win.loc[data_win["단계"]==32,'승리팀_포메이션'].value_counts().add(data_win.loc[data_win["단계"]==32,'패배팀_포메이션'].value_counts(),fill_value=0)
data_win.loc[data_win["단계"]==32,'승리팀_포메이션'].value_counts().div(x,fill_value=0).round(2)*100

df=pd.concat([x,data_win.loc[data_win["단계"]==32,'승리팀_포메이션'].value_counts().div(x,fill_value=0).round(2)*100],axis=1)

## 조별리그 단계에서 승률이 높은 포메이션의 빈도와 승률
df.columns=["빈도","승률"]
df
