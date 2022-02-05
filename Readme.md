# MARL methods in mpe environment
This is a record for my learning process during Multi-Agent Reinforcement Learning algorithms. 

## Requirement
- python 3.8
- pytorch 1.8.2

## Instructions  
Download the multiagent-particle-environment from github and open a terminal then `cd` into the root directory (where setup.py exists). Then use command  `python setup.py install`. Then in a new python script, try import `multiagent`.  

## Possible errors
- pytorch:  `cannot import name 'Final' from 'typing'`. Probabily your python version is 3.7, change it to 3.8.  
- gym: `cannot import name 'prng' from gym.spaces`. prng package is deleted. Install gym==0.10.5, or edit in the file as in [2]. 



## References
-[1] mpe environment :
> https://github.com/openai/multiagent-particle-envs  

-[2] To fix 'prng' lost problem  
> https://github.com/openai/multiagent-particle-envs/issues/53  

-[3] a great example repo:
> https://github.com/AI4Finance-Foundation/ElegantRL
 
-[4] Morphan's example (for single agent rl)
> https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
