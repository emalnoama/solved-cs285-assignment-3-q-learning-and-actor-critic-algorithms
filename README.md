Download Link: https://assignmentchef.com/product/solved-cs285-assignment-3-q-learning-and-actor-critic-algorithms
<br>
<h1>1           Part 1: Q-Learning</h1>

<h2>1.1         Introduction</h2>

Part 1 of this assignment requires you to implement and evaluate Q-learning with convolutional neural networks for playing Atari games. The Q-learning algorithm was covered in lecture, and you will be provided with starter code. This assignment will be faster to run on a GPU, though it is possible to complete on a CPU as well. We recommend using the Colab option if you do not have a GPU available to you. Please start early!

<h2>1.2         File overview</h2>

The starter code for this assignment can be found at

<a href="https://github.com/berkeleydeeprlcourse/homework_fall2020/tree/master/hw3">https://github.com/berkeleydeeprlcourse/homework_fall2020/tree/master/hw3</a>

We will be building on the code that we have implemented in the first two assignments. All files needed to run your code are in the hw3 folder, but there will be some blanks you will fill with your solutions from homework 1. These locations are marked with # TODO: get this from hw1 or hw2 and are found in the following files:

<ul>

 <li>infrastructure/rl trainer.py</li>

 <li>infrastructure/utils.py</li>

 <li>policies/MLP policy.py</li>

</ul>

In order to implement deep Q-learning, you will be writing new code in the following files:

<ul>

 <li>agents/dqn py</li>

 <li>critics/dqn critic.py</li>

 <li>policies/argmax policy.py</li>

</ul>

There are two new package requirements (opencv-python and gym[atari]) beyond what was used in the first two assignments; make sure to install these with pip install -r requirements.txt if you are running the assignment locally.

<h2>1.3         Implementation</h2>

The first phase of the assignment is to implement a working version of Q-learning. The default code will run the Ms. Pac-Man game with reasonable hyperparameter settings. Look for the # TODO markers in the files listed above for detailed implementation instructions. You may want to look inside infrastructure/dqn utils.py to understand how the (memory-optimized) replay buffer works, but you will not need to modify it.

Once you implement Q-learning, answering some of the questions may require changing hyperparameters, neural network architectures, and the game, which should be done by changing the command line arguments passed to run_hw3_dqn.py or by modifying the parameters of the Args class from within the Colab notebook.

To determine if your implementation of Q-learning is correct, you should run it with the default hyperparameters on the Ms. Pac-Man game for 1 million steps using the command below. Our reference solution gets a return of 1500 in this timeframe. On Colab, this will take roughly 3 GPU hours. If it takes much longer than that, there may be a bug in your implementation.

To accelerate debugging, you may also test on LunarLander-v3, which trains your agent to play Lunar Lander, a 1979 arcade game (also made by Atari) that has been implemented in OpenAI Gym. Our reference solution with the default hyperparameters achieves around 150 reward after 350k timesteps, but there is considerable variation between runs and without the double-Q trick the average return often decreases after reaching 150.

We recommend using LunarLander-v3 to check the correctness of your code before running longer experiments with MsPacman-v0.

<h2>1.4         Evaluation</h2>

Once you have a working implementation of Q-learning, you should prepare a report. The report should consist of one figure for each question below. You should turn in the report as one PDF and a zip file with your code. If your code requires special instructions or dependencies to run, please include these in a file called README inside the zip file.

<strong>Question 1: basic Q-learning performance. (DQN) </strong>Include a learning curve plot showing the performance of your implementation on Ms. Pac-Man. The x-axis should correspond to number of time steps (consider using scientific notation) and the y-axis should show the average per-epoch reward as well as the best mean reward so far. These quantities are already computed and printed in the starter code. They are also logged to the data folder, and can be visualized using Tensorboard as in previous assignments. Be sure to label the y-axis, since we need to verify that your implementation achieves similar reward as ours. You should not need to modify the default hyperparameters in order to obtain good performance, but if you modify any of the parameters, list them in the caption of the figure. The final results should use the following experiment name:

python cs285/scripts/run_hw3_dqn.py –env_name MsPacman-v0 –exp_name q1

<strong>Question 2: double Q-learning (DDQN). </strong>Use the double estimator to improve the accuracy of your learned Q values. This amounts to using the online Q network (instead of the target Q network) to select the best action when computing target values. Compare the performance of DDQN to vanilla DQN. Since there is considerable variance between runs, you must run at least three random seeds for both DQN and DDQN. You may uuse LunarLander-v3 for this question. The final results should use the following experiment names:

python cs285/scripts/run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q2_dqn_1 –seed 1 python cs285/scripts/run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q2_dqn_2 –seed 2 python cs285/scripts/run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q2_dqn_3 –seed 3

python cs285/scripts/run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q2_doubledqn_1 -double_q –seed 1 python cs285/scripts/run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q2_doubledqn_2 -double_q –seed 2 python cs285/scripts/run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q2_doubledqn_3 -double_q –seed 3

Submit the run logs (in cs285/data) for all of the experiments above. In your report, make a single graph that averages the performance across three runs for both DQN and double DQN. See scripts/read results.py for an example of how to read the evaluation returns from Tensorboard logs.

<strong>Question 3: experimenting with hyperparameters. </strong>Now let’s analyze the sensitivity of Q-learning to hyperparameters. Choose one hyperparameter of your choice and run at least three other settings of this hyperparameter, in addition to the one used in Question 1, and plot all four values on the same graph. Your choice what you experiment with, but you should explain why you chose this hyperparameter in the caption. Examples include: learning rates, neural network architecture, exploration schedule or exploration rule (e.g. you may implement an alternative to -greedy), etc. Discuss the effect of this hyperparameter on performance in the caption. You should find a hyperparameter that makes a nontrivial difference on performance. Note: you might consider performing a hyperparameter sweep for getting good results in Question 1, in which case it’s fine to just include the results of this sweep for Question 3 as well, while plotting only the best hyperparameter setting in Question 1. The final results should use the following experiment name:

python run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q3_hparam1

python run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q3_hparam2

python run_hw3_dqn.py –env_name LunarLander-v3 –exp_name q3_hparam3

You can replace LunarLander-v3 with PongNoFrameskip-v4 or MsPacman-v0 if you would like to test on a different environment.

<h1>2           Part 2: Actor-Critic</h1>

<h2>2.1         Introduction</h2>

Part 2 of this assignment requires you to modify policy gradients (from hw2) to an actor-critic formulation. Part 2 is relatively shorter than part 1. The actual coding for this assignment will involve less than 20 lines of code. Note however that evaluation may take longer for actor-critic than policy gradient (on half-cheetah) due to the significantly larger number of training steps for the value function.

Recall the policy gradient from hw2:

<em>.</em>

In this formulation, we estimate the Q function by taking the sum of rewards to go over each trajectory, and we bustract the value function baseline to obtain the advantage

In practice, the estimated advantage value suffers from high variance. Actor-critic addresses this issue by using a <em>critic network </em>to estimate the sum of rewards to go. The most common type of critic network used is a value function, in which case our estimated advantage becomes

In this assignment we will use the same value function network from hw2 as the basis for our critic network. One additional consideration in actor-critic is updating the critic network itself. While we can use Monte Carlo rollouts to estimate the sum of rewards to go for updating the value function network, in practice we fit our value function to the following <em>target values</em>:

<em>y<sub>t </sub></em>= <em>r</em>(<em>s<sub>t</sub>,a<sub>t</sub></em>) + <em>γV <sup>π</sup></em>(<em>s<sub>t</sub></em><sub>+1</sub>)

we then regress onto these target values via the following regression objective which we can optimize with gradient descent:

minX(<em>V</em><em>φπ</em>(<em>s</em><em>it</em>) − <em>y</em><em>it</em>)2

<em>φ</em>

<em>i,t</em>

In theory, we need to perform this minimization every time we update our policy, so that our value function matches the behavior of the new policy. In practice however, this operation can be costly, so we may instead just take a few gradient steps at each iteration. Also note that since our target values are based on the old value function, we may need to recompute the targets with the updated value function, in the following fashion:

<ol>

 <li>Update targets with current value function</li>

 <li>Regress onto targets to update value function by taking a few gradient steps</li>

 <li>Redo steps 1 and 2 several times</li>

</ol>

In all, the process of fitting the value function critic is an iterative process in which we go back and forth between computing target values and updating the value function to match the target values. Through experimentation, you will see that this iterative process is crucial for training the critic network.

<h2>2.2         Implementation</h2>

Your code will build off your solutions from homework 2. You will need to fill in the TODOS for the following parts of the code. To run the code, go into the hw3 directory and simply execute python cs285/scripts/run_hw3_actor_criti

<ul>

 <li>In policies/MLP_policy.py copy over your policy class for PG, you should note that the AC policy class is in fact the same as the policy class you implemented in the policy gradient homework (except we no longer have a nn baseline).</li>

 <li>In agents/ac_agent.py, finish the train This function should implement the necessary critic updates, estimate the advantage, and then update the policy. Log the final losses at the end so you can monitor it during training.</li>

 <li>In agents/ac_agent.py, finish the estimate_advantage function: this function uses the critic network to estimate the advantage values. The advantage values are computed according to</li>

</ul>

Note: for terminal timesteps, you must make sure to cut off the reward to go (i.e., set it to zero), in which case we have

<ul>

 <li>critics/bootstrapped_continuous_critic.py complete the TODOS in update. In update, perform the critic update according to process outlined in the introduction. You must perform num_grad_steps_per_target_update * self.num_target_updates number of updates, and recompute the target values every self.num_grad_steps_per_target_update number of steps.</li>

</ul>

<h2>2.3         Evaluation</h2>

Once you have a working implementation of actor-critic, you should prepare a report. The report should consist of figures for the question below. You should turn in the report as one PDF (same PDF as part 1) and a zip file with your code (same zip file as part 1). If your code requires special instructions or dependencies to run, please include these in a file called README inside the zip file.

<strong>Question 4: Sanity check with Cartpole  </strong>Now that you have implemented actor-critic, check that your solution works by running Cartpole-v0.

<table width="634">

 <tbody>

  <tr>

   <td width="634">python run_hw3_actor_critic.py –env_name CartPole-v0 -n 100 -b 1000 –exp_name q4_ac_1_1-ntu 1 -ngsptu 1</td>

  </tr>

 </tbody>

</table>

In the example above, we alternate between performing one target update and one gradient update step for the critic. As you will see, this probably doesn’t work, and you need to increase both the number of target updates and number of gradient updates. Compare the results for the following settings and report which worked best. Do this by plotting all the runs on a single plot and writing your takeaway in the caption.

python run_hw3_actor_critic.py –env_name CartPole-v0 -n 100 -b 1000 –exp_name q4_100_1 ntu 100 -ngsptu 1

python run_hw3_actor_critic.py –env_name CartPole-v0 -n 100 -b 1000 –exp_name q4_1_100 ntu 1 -ngsptu 100

python run_hw3_actor_critic.py –env_name CartPole-v0 -n 100 -b 1000 –exp_name q4_10_10 ntu 10 -ngsptu 10

At the end, the best setting from above should match the policy gradient results from Cartpole in hw2 (200).

<strong>Question 5: Run actor-critic with more difficult tasks </strong>Use the best setting from the previous question to run InvertedPendulum and HalfCheetah:

python run_hw3_actor_critic.py –env_name InvertedPendulum-v2 –ep_len 1000 –discount

0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 –exp_name q5_&lt;ntu&gt;_&lt;ngsptu&gt; -ntu &lt;&gt; -ngsptu &lt;&gt;

where &lt;ntu&gt; &lt;ngsptu&gt; is replaced with the parameters you chose.

python run_hw3_actor_critic.py –env_name HalfCheetah-v2 –ep_len 150 –discount 0.90 -scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 –exp_name q5_&lt;ntu&gt;_&lt; ngsptu&gt; -ntu &lt;&gt; -ngsptu &lt;&gt;

Your results should roughly match those of policy gradient. After 150 iterations, your HalfCheetah return should be around 150 and your InvertedPendulum return should be around 1000. Your deliverables for this section are plots with the eval returns for both enviornments.

As a debugging tip, the returns should start going up immediately. For example, after 20 iterations, your HalfCheetah return should be above -40 and your InvertedPendulum return should near or above 100. However, there is some variance between runs, so the 150-iteration results is the more important number.

<h1>3           Submitting the code and experiment runs</h1>

In order to turn in your code and experiment logs, create a folder that contains the following:

<ul>

 <li>A folder named data with all the experiment runs from this assignment. <strong>Do not change the names originally assigned to the folders, as specified by </strong>exp name <strong>in the instructions. Video logging is disabled by default in the code, but if you turned it on for debugging, you will need to run those again with </strong>–video log freq -1<strong>, or else the file size will be too large for submission.</strong></li>

 <li>The cs285 folder with all the .py files, with the same names and directory structure as the original homework repository (excluding the data folder). Also include any special instructions we need to run in order to produce each of your figures or tables (e.g. “run python myassignment.py -sec2q1” to generate the result for Section 2 Question 1) in the form of a README file.</li>

</ul>

As an example, the unzipped version of your submission should result in the following file structure. <strong>Make sure that the submit.zip file is below 15MB and that they include the prefix </strong>q1 <strong>, </strong>q2 <strong>, </strong>q3 <strong>, etc. </strong>agents

… …

Turn in your assignment on Gradescope. Upload the zip file with your code and log files to <strong>HW3 Code</strong>, and upload the PDF of your report to <strong>HW3</strong>.