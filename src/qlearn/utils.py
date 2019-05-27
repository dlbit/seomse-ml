import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#데이터 폴더에서 데이터를 얻어온다.
def get_data(col='close'):
  """ Returns a 3 x n_step array """
  msft = pd.read_csv('data/daily_MSFT.csv', usecols=[col])
  ibm = pd.read_csv('data/daily_IBM.csv', usecols=[col])
  qcom = pd.read_csv('data/daily_QCOM.csv', usecols=[col])
  # recent price are at top; reverse it
  return np.array([msft[col].values[::-1],
                   ibm[col].values[::-1],
                   qcom[col].values[::-1]])

  # 스케일링 수행.
  # 스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에 선형 변환을 적용하여
  # 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다.
  # 스케일링은 자료의 오버플로우나 언더플로우를 방지하고 독립 변수의 공분산
  # 행렬의 조건수를 감소시켜 최적화 과정에서의 안정성 및 수렴 속도를 향상시킨다.
def get_scaler(env):
  # env를 구하고 그 observation space를 스케일링 해서 돌려준다
  low = [0] * (env.n_stock * 2 + 1)

  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_cash = env.init_invest * 3 # 3 is a magic number...
  max_stock_owned = max_cash // min_price
  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  high.append(max_cash)

  scaler = StandardScaler() #sklearn
  scaler.fit([low, high])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)