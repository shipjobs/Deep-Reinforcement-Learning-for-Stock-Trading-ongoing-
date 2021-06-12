#from Util import *
#from DQN import *

#######################################################################
# 과제: Deep Reinforcement Learning on Stock Data
#######################################################################
# reference : https://www.kaggle.com/itoeiji/deep-reinforcement-learning-on-stock-data


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#######################################################################
import time
import copy
import numpy as np
import pandas as pd

#import chainer       ##딥러닝 프레임워크  ->
#import chainer.functions as F
#import chainer.links as L

#기존 chainer 을 torch 로 변환
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl


from subprocess import check_output
from RL_model.DQN import train_dqn
from env import trader_env

import util

#print(check_output(["ls", "/home/ubuntu/USER_SKKU/leeinsung/RL_team_project/input"]).decode("utf8"))

#######################################################################
#init_notebook_mode()  #쥬피터 노트북 모드 켜기.. (필요?)

#data = pd.read_csv("/home/ubuntu/USER_SKKU/leeinsung/RL_team_project/input/Data/Stocks/a.us.txt")
################## 상대 경로
# 임시로 절대 경로.. 지정
# 임의의 주식 데이터 가져 오기.
data = pd.read_csv("/home/ubuntu/USER_SKKU/leeinsung/RL_team_project/input/Data/Stocks/aapl.us.txt")  

data['Date'] = pd.to_datetime(data['Date'])

data = data.set_index('Date')
print(data.index.min(), data.index.max())
data.head()

#print(data)
#######################################################################
date_split = '2016-01-01'
train = data[:date_split]
test = data[date_split:]

len(train), len(test)

#######################################################################
#
Q, total_losses, total_rewards = train_dqn(trader_env(train))   
plot_train_test_by_q(trader_env(train), trader_env(test), Q, 'DQN')

#######################################################################
#
"""
plot_train_test(train, test, date_split)
"""
#######################################################################
"""
env = Environment1(train)
print(env.reset())
for _ in range(3):
    pact = np.random.randint(3)
    print(env.step(pact))
"""
"""
#가시화
plot_loss_reward(total_losses, total_rewards)
"""

"""
#가시화
Q, total_losses, total_rewards = train_ddqn(Environment1(train))
plot_loss_reward(total_losses, total_rewards)
plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'Double DQN')
"""
 
