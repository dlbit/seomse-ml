import gym # 오픈된 강화학습 라이브러리 openai
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv(gym.Env):
  """
  data 폴더의 IBM , MSFT , QCOM 의 주가 파일로 학습을 함.

  State (상태는 3가지로 정의.): [# of stock owned, current stock prices, 현금을 보유하고 있음]
    - 배열의 개수는 n_stock (학습데이터) * 2 + 1
    - 가격을 정수로 이산화 해서 계산
    - 각 주식에 종가 가격을 사용함
    - 수행중인 행동을 토대로 각 단계에서 현금을 계산함

  Action (행위) : 판매 (0), 홀딩 (1), 구매 (2)
    - 판매시 , 가진 물량을 전부 판매함.
    - 구매시 , 가진 현금을 모두 사용함
    - 여러 주식을 구매시 , 현금을 주식수만큼 똑같이 분배 한 후 구매함
  """
  def __init__(self, train_data, init_invest=20000):
    # 데이터
    self.stock_price_history = np.around(train_data) # 데이터를 0.5 기준으로 올림 혹은 내림 처리
    self.n_stock, self.n_step = self.stock_price_history.shape   # 학습데이터가 몇개의 행렬을 가지고 있는지 체크한다.

    # 인스턴스 변수들
    self.init_invest = init_invest
    self.cur_step = None # None == null
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    # action space / 에이젼트에게 전달할수 있는 명령 공간.
    self.action_space = spaces.Discrete(3**self.n_stock) # ** 거듭제곱. 이 소스에서는 3의 3제곱 = 27

    # observation space: 스케일러 샘플링 및 구축을 위한 공간
    stock_max_price = self.stock_price_history.max(axis=1)
    stock_range = [
                    [0, init_invest * 2 // mx]
                    for mx in stock_max_price
                  ] # // => 몫 계산
    price_range = [[0, mx] for mx in stock_max_price]
    cash_in_hand_range = [[0, init_invest * 2]]
    self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)

    # seed and start
    self._seed()
    self._reset()


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def _reset(self):
    self.cur_step = 0
    self.stock_owned = [0] * self.n_stock
    self.stock_price = self.stock_price_history[:, self.cur_step]
    self.cash_in_hand = self.init_invest
    return self._get_obs()


  def _step(self, action):
    assert self.action_space.contains(action)
    prev_val = self._get_val()
    self.cur_step += 1
    self.stock_price = self.stock_price_history[:, self.cur_step] # update price
    self._trade(action)
    cur_val = self._get_val()
    reward = cur_val - prev_val
    done = self.cur_step == self.n_step - 1
    info = {'cur_val': cur_val}
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    obs = []
    obs.extend(self.stock_owned)
    obs.extend(list(self.stock_price))
    obs.append(self.cash_in_hand)
    return obs


  def _get_val(self):
    return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand


  def _trade(self, action):
    # all combo to sell(0), hold(1), or buy(2) stocks
    action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
    action_vec = action_combo[action]

    # one pass to get sell/buy index
    sell_index = []
    buy_index = []
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    # two passes: sell first, then buy; might be naive in real-world settings
    if sell_index:
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
    if buy_index:
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1 # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False
