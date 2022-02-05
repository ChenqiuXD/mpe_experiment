# MARL methods in mpe environment
This is a record for my learning process during Multi-Agent Reinforcement Learning algorithms. 

## Instructions  
Download the multiagent-particle-environment from github and open a terminal then `cd` into the root directory (where setup.py exists). Then use command  
> python setup.py install   

to install the mpe environment.  Furthermore, note that since it require gym==0.10.5, yet this is an old version. Refer to the [2] website to fix the problem  
> ImportError: cannot import name 'prng' from 'gym.spaces'

## References
-[1] mpe environment :
> https://github.com/openai/multiagent-particle-envs  
-[2] To fix 'prng' lost problem  
> https://github.com/openai/multiagent-particle-envs/issues/53  
-[3] a great example repo:
> https://github.com/AI4Finance-Foundation/ElegantRL
-[4] Morphan's example (for single agent rl)
> https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
