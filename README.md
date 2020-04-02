[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

### Environment Description
#### States
* The observation space consists of 8 dimensions.
* It is corresponding to the position and velocity of the ball and racket.
* Two agents control rackets to bounce a ball over a net. Each agent receives its own, local observation.

#### Actions
* Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

#### Rewards
* An agent receives a reward of +0.1, if an agent hits the ball over the net.
* If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

#### Goal
* The goal of the two agents is to collaborate to keep the ball in play.
* The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5. 

###  Project Structure
The repository contains the following files.
* Tennis.ipynb Contains the agent training code for Unity Tennis environment.
* maddpg_agent.py Contains MADDPG based agent implemenation.
* model.py Contains actor and critic network.
* noise.py Contians Ornstein-Uhlenbeck noise process utility class.
* replay_buffer.py Contains replay buffer utility class.
* train.py Contains training utility methods.
* main.py It is an entry file for training in normal python way. It is an alternative to Tennis.ipynb. 
* checkpoint_actor_0/1.pth and checkpoint_critic_0/1.pth are pre-trained model parameters' file.
* Report.ipynb Contains project write-up or report, which details the implementation and algorithm.

### Getting Started
1. Install Anaconda(https://conda.io/docs/user-guide/install/index.html)
2. Install dependencies by issue:
```
pip install -r requirements.txt
```
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Place the file in the root folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training, 
or directly jump to **Watch Smart Agent** using pre-trained weights, checkpoint_actor_0/1.pth and checkpoint_critic_0/1.pth, 
to watch the performance of the two trained agents.  
