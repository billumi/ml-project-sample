
import numpy as np
def q_learning(env, episodes=100):
    q=np.zeros([env.observation_space.n, env.action_space.n])
    for _ in range(episodes):
        s=env.reset(); done=False
        while not done:
            a=env.action_space.sample()
            ns,r,done,_=env.step(a)
            q[s,a]=q[s,a]+0.1*(r+0.6*np.max(q[ns])-q[s,a])
            s=ns
    return q
