from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam



def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
  # Q 함수로 쓰이는 Multi-layer Perceptron이 구현되어 있다
  #keras쪽 스터디 필요
  model = Sequential()
  model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
  for _ in range(n_hidden_layer):
    model.add(Dense(n_neuron_per_layer, activation=activation))
  model.add(Dense(n_action, activation='linear'))
  model.compile(loss=loss, optimizer=Adam())
  print(model.summary())
  return model