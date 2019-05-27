import pickle # 객체구조의 직렬화 + 역직렬화; 모델 저장 용도
import time # 운영체제등의 시간을 다룸
import numpy as np # 행렬 연산을 위한 라이브러리, 많이 사용
import argparse # 프로그램 실행 명령을 파싱하는 라이브러리
import re # Regex , 정규표현식

from src.qlearn.envs import TradingEnv
from src.qlearn.agent import DQNAgent
from src.qlearn.utils import get_data, get_scaler, maybe_make_dir



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode', type=int, default=2000,
                      help='number of episode to run')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
  parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                      help='initial investment amount')
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
  args = parser.parse_args()

  maybe_make_dir('weights')
  maybe_make_dir('portfolio_val')

  timestamp = time.strftime('%Y%m%d%H%M')

  data = np.around(get_data())
  train_data = data[:, :3526]
  test_data = data[:, 3526:]

  env = TradingEnv(train_data, args.initial_invest)
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  portfolio_value = []

  if args.mode == 'test':
    # env를 테스트 데이터로 재 생성
    env = TradingEnv(test_data, args.initial_invest)
    # arg로 전달받은 weights 값을 로드한다. 미리 학습 되어 있어야 함.
    agent.load(args.weights)
    # 파일 생성을 위해 시간을 기록
    timestamp = re.findall(r'\d{12}', args.weights)[0]

  for e in range(args.episode): # episode = 기본값 2000회
    state = env.reset()
    state = scaler.transform([state]) # sklean 부분
    for time in range(env.n_step):
      action = agent.act(state)
      next_state, reward, done, info = env.step(action)
      next_state = scaler.transform([next_state])
      if args.mode == 'train':
        agent.remember(state, action, reward, next_state, done)
      state = next_state
      if done:
        print("episode: {}/{}, episode end value: {}".format(
          e + 1, args.episode, info['cur_val']))
        portfolio_value.append(info['cur_val']) # 포트폴리오 결과를 에피소드에 추가함
        break
      if args.mode == 'train' and len(agent.memory) > args.batch_size:
        agent.replay(args.batch_size)
    if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
      agent.save('weights/{}-dqn.h5'.format(timestamp))

  # 디스크에 값 저장.
  with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
    pickle.dump(portfolio_value, fp)