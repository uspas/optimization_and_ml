import numpy as np
import math
import gym
import torch


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    
    '''
    inputs an environment, agent, max episodes, max steps, and batch size
    
    trains agent in the environment
    '''
    
    episode_rewards = []

    for episode in range(max_episodes):
        
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            
            action = agent.get_action(state)
            #action = agent.get_noisy_action(state,step)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                #state = env.reset()
                print("Episode " + str(episode) + ":  " + str(episode_reward))
                print("---------")
                break
                
            state = next_state

    return episode_rewards
