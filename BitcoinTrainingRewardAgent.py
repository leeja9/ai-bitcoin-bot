# TODO: Define reward behavior based on environment state
class BitcoinTrainingRewardAgent:
    def __init__(self) -> None:
        pass


# Reference:
#   Multi-Objective reward generalization: Improving performance of
#       Deep Reinforcement Learning for applications in single-asset trading
#   By: Federico Cornalba, Constantin Disselkamp,
#       Davide Scassola, Christopher Helf
#   URL: https://arxiv.org/abs/2203.04579
#   Companion repo: https://github.com/trality/fire

# ALGORITHM 1 PSEUDOCODE

# Algorithm 1 (DQN-HER)
# Input: MultiReward ∈ {True, False}
# Parameters: tol ∈ (0, 1), batchsize, 𝑘,𝑀 ∈ N, 𝑆 ⊂ {1, . . . ,𝑀}
# Output: Trained Multi-Reward agent.
# 1: Take one-hot encoding vector 𝒘.
# 2: Initialize network 𝑄 : (𝒔,𝒘) ↦→ [𝑄1(𝒔,𝒘), . . . ,𝑄𝑃 (𝒔,𝒘)] mapping
# state variables 𝒔 and weight vector 𝒘 to expected discounted
# return of every action 𝑎 ∈ {1, . . . , 𝑃}.
# 3: for 𝑖 = 1, . . . ,𝑀 do
# 4: Reset training environment.
# 5: while Episode 𝑖 not finished do
# 6: if MultiReward is True then
# 7: Sample random reward weights 𝒘.
# 8: end if
# 9: if Unif(0, 1) < 𝑡𝑜𝑙 then
# 10: Choose random action 𝑎
# 11: else
# 12: Pick 𝑎 ← arg max𝑎 𝑄(𝒔,𝒘).
# 13: end if
# 14: Conduct one step (get to new state 𝒔𝑛𝑒𝑤)
# 15: Get associated reward 𝒘 · 𝒓 (𝒔, 𝑎)
# 16: Append single experience (𝒔,𝒘,𝒘 · 𝒓 (𝒔, 𝑎), 𝑎, 𝒔𝑛𝑒𝑤) to experience
# replay R
# 17: Set 𝒔 ← 𝒔𝑛𝑒𝑤
# 18: if MultiReward is True then
# 19: Add other 𝑘 experiences to Replay by re-running lines
# 7-14-15-16 𝑘 times.
# 20: end if
# 21: if 𝑖 ∈ 𝑆 then
# 22: Randomly sample batchsize units from R.
# 23: Update 𝑄-values associated with each sampled unit
# using Bellman equation (2).
# 24: Fit model on estimated 𝑄-values
# 25: end if
# 26: end while
# 27: end for

# ALGORITHM 2 PSEUDOCODE

# Algorithm 2 (DQN-HER) with discount factor generalization and
# random access point (Changes to lines of Algorithm 1)
# Input: Anchoring for cross-validation ∈ {True; False}
# 2: Initialize network 𝑄 : (𝒔,𝒘,𝛾) ↦→ [𝑄1(𝒔,𝒘,𝛾), . . . ,𝑄𝑃 (𝒔,𝒘,𝛾)]
# mapping state variables 𝒔, weight vector 𝒘, and discount factor
# 𝛾 to expected discounted return of every action 𝑎 ∈ {1, . . . , 𝑃}.
# ...
# 4: if not Anchoring for cross-validation then
# 5: Randomly select subset of training set, and reset associated
# environment.
# 6: end if
# ...
# 7: Sample random reward weights 𝒘 and discount factor 𝛾.
# ...
# 12: Pick 𝑎 ← arg max𝑎 𝑄(𝒔,𝒘,𝛾).
# ...
# 16: Append single experience (𝒔,𝛾,𝒘,𝒘 · 𝒓 (𝒔, 𝑎), 𝑎, 𝒔𝑛𝑒𝑤) to experience
# replay R
# ...
# 22: Randomly sample batchsize units from R, and normalize sampled 𝑄-values
# according to Subsection 4.2.1.
