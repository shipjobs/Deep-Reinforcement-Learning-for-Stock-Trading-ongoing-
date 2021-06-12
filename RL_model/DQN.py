import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Q_Network(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def dqn_train(env):

    Q = Q_Network(input_size=env.history_len+1, hidden_size=100, output_size=3)
    Q_copy = copy.deepcopy(Q)
    optimizer = optim.Adam(Q.parameters())

    num_episodes = 50
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 20
    epsilon = 1.0
    epsilon_decay = 1e-3
    epsilon_min = 0.1
    epsilon_decay_start = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    seed =1 
    np.random.seed(seed)
    torch.manual_seed(seed)

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    print('DQN Training')
    for epoch in range(num_episodes):
        print('epoch:',epoch,'\n')

        state = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:
            print('step:',step,end='\r',flush=True)

            # select act
            action = torch.randint(0,3,(1,))
            if torch.rand(1, dtype=torch.float32) > epsilon:
                action = Q(torch.tensor(state, dtype=torch.float32).reshape(1, -1))
                action = torch.argmax(action.data)

            # act
            next_state, reward, done = env.step(action)

            # add memory
            memory.append((state, action, reward, next_state, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = memory.copy()
                    random.shuffle(shuffled_memory)
                    memory_idx = range(len(shuffled_memory))
                    
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_state = torch.tensor(batch[:, 0].tolist(), dtype=torch.float32).reshape(batch_size, -1)
                        b_action = torch.tensor(batch[:, 1].tolist(), dtype=torch.int32)
                        b_reward = torch.tensor(batch[:, 2].tolist(), dtype=torch.int32)
                        b_nextstate = torch.tensor(batch[:, 3].tolist(), dtype=torch.float32).reshape(batch_size, -1)
                        b_done = torch.tensor(batch[:, 4].tolist(), dtype=torch.bool)

                        q = Q(b_state)
                        maxq = torch.max(Q_copy(b_nextstate).data, axis=1)
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            temp = b_action[j].item()
                            target[j, temp] = b_reward[j]+gamma*maxq.values[j]*(not b_done[j])
                        optimizer.zero_grad()       
                        MSE = nn.MSELoss()
                        loss = MSE(q, target)
                        total_loss += loss
                        loss.backward()
                        optimizer.step()

                if total_step % update_q_freq == 0:
                    Q_copy = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > epsilon_decay_start:
                epsilon -= epsilon_decay

            # next step
            total_reward += reward
            state = next_state
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('reward : %0.3f' %(log_reward),'loss : %0.3f' %(log_loss.item()),'time: %0.1f s' %(elapsed_time))
            start = time.time()
            
    return Q, total_losses, total_rewards