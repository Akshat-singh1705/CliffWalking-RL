# CliffWalking: SARSA vs Q-Learning

A Reinforcement Learning project implementing and comparing **SARSA and Q-Learning** on the CliffWalking environment using Python. This project explores how on-policy and off-policy temporal-difference learning algorithms learn to navigate an environment, avoid the cliff, and reach the goal.

## 📌 Project Overview

The CliffWalking environment consists of a grid in which an agent must navigate from a starting position to a goal while avoiding a dangerous cliff. The agent receives rewards based on its actions and learns a policy through interaction with the environment.

This project implements two reinforcement learning algorithms:

- **SARSA:** An on-policy temporal-difference learning algorithm.
- **Q-Learning:** An off-policy temporal-difference learning algorithm.

Both algorithms use Q-tables and an epsilon-greedy strategy to balance exploration and exploitation.

## 🧠 Algorithms Implemented

### 1. SARSA

SARSA stands for State–Action–Reward–State–Action. It updates the Q-value using the action selected by the current policy in the next state.

**Update equation:**

\[
Q(s,a)\leftarrow Q(s,a)+\alpha
[r+\gamma Q(s',a')-Q(s,a)]
\]

### 2. Q-Learning

Q-Learning updates the Q-value using the maximum estimated Q-value of the next state, independently of the action actually selected by the behavior policy.

**Update equation:**

\[
Q(s,a)\leftarrow Q(s,a)+\alpha
[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
\]

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---:|
| Learning rate (\(\alpha\)) | 0.5 |
| Discount factor (\(\gamma\)) | 0.99 |
| Exploration rate (\(\epsilon\)) | 0.1 |
| Training episodes | 500 |
| Number of states | 48 |
| Number of actions | 4 |

## 📊 Performance Comparison

After training, both agents were evaluated over 500 episodes. The following results are based on the recorded outputs from the notebooks.

| Metric | SARSA | Q-Learning |
|---|---:|---:|
| Average reward | -24.038 | -47.354 |
| Average steps | 20.474 | 16.466 |

### Observations

- **SARSA achieved a better average reward** in these experiments, with a less negative score.
- **Q-Learning took fewer steps on average**, despite receiving a worse average reward.
- The difference between reward and step count shows why both metrics are useful when evaluating a learned policy.

These results reflect the current implementations, training settings, and evaluation procedures. They should not be interpreted as a definitive ranking of the algorithms without checking that both experiments use comparable evaluation conditions.

## 📈 Learning Curves

The learning curves visualize the total reward collected during training and help illustrate how each algorithm's performance changes across episodes.

### SARSA Learning Curve

![SARSA Learning Curve](images/SARSA_Learning_Curve.jpg)

### Q-Learning Learning Curve

![Q-Learning Learning Curve](images/QLearning_Learning_Curve.jpg)

## 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib
- Gymnasium
- Jupyter Notebook

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the URL with your actual GitHub repository URL.

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

```bash
jupyter notebook
```

Open either notebook:

- `CliffWalking_SARSA.ipynb`
- `CliffWalking_Qlearning.ipynb`

Run the notebook cells to train the corresponding agent, visualize the environment where enabled, and evaluate the learned policy.

**Note:** Running the training cells again will retrain the agent. Results may differ depending on random initialization and the exploration process.

## 📂 Repository Structure

```text
CliffWalking-RL/
│
├── CliffWalking_SARSA.ipynb
├── CliffWalking_Qlearning.ipynb
├── requirements.txt
├── README.md
└── images/
    ├── SARSA_Learning_Curve.jpg
    └── QLearning_Learning_Curve.jpg
```


## 🎯 Learning Outcomes

- Understanding reinforcement learning fundamentals.
- Implementing tabular SARSA and Q-Learning.
- Understanding on-policy versus off-policy learning.
- Applying epsilon-greedy exploration.
- Tracking cumulative rewards and episode lengths.
- Comparing reinforcement learning algorithms using empirical results.
