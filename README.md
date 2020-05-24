# Atari-Deep-Reinforcement-Learning-

This Repository is part of our NNFL Course Project(Paper ID - 83). 

# Group Members :

* Rohit Bohra - 2017A7PS0225P
* Fauzaan Qureshi - 2017A2PS0663P
* Kushaghra Raina - 2017A7PS0161P

We have implemented the algorithm(Deep Q-learning with Experience Replay) given in this novel paper.
Paper Link - https://arxiv.org/pdf/1312.5602.pdf

# Deep Q-learning with Experience Replay :

# Implementation Details :
We have used Keras for all implementations and Matplotlib to visualize the graphs for rewards and losses. We have implemented the same alogrithm mentioned above. We had previously trained using both optimizers - RMSProp and Adam. Adam worked better for our implementation. Replay Memory Size was fixed at 40000 experiences because of ram limitations. Before training, initial exploration is done to gain some random experiences and fill up our replay memory. Our implementation can be used to train any atari game. We trained **Breakout** and **SpaceInvaders** in a deterministic enviroment using OpenAI Gym library. After training for _ episodes, we started getting satisfying results. Training for SpaceInvaders is still in progress but we hope to complete it soon with good results.

# Conclusions from the experiments :

# Common Issues faced by us :

* We need to take care of some issues in games like Breakout where fire has to be pressed manually after losing a life. This is done because the Q-value of "fire" becomes very less and it is very hard to determine those 5 occurences(5 Lives) of pressing fire. There is a similar problem in Pong as well.

* Initially in Breakout, our agent was getting decent score but it was because our agent was getting stuck at local minima. The Slider moved to the extreme of the frame to gain some initial advantage but was getting stuck there.

# Environment Details :
Please refer to our [requirements.txt][https://github.com/geeky-wizard/Atari-Deep-Reinforcement-Learning-/blob/master/requirements.txt] file to get the enviroment details.

# Instructions to Run :
