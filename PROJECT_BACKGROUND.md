# LLM-RL-Tuner Project Background & Knowledge Base

## 📋 Project Overview

**Goal**: Implement an end-to-end AutoRL system that combines reward function optimization, hyperparameter tuning, and algorithm selection using Large Language Models.

**Key Innovation**: First integrated framework combining Eureka + AgentHPO capabilities with Algorithm Selection, without requiring complex simulation environments like Isaac Gym.

---

## 🎯 What We're Building

### Core System: LLM-RL-Tuner

**Input**:
- Environment (Gymnasium-compatible)
- Task description (natural language)

**Automated Process**:
1. **Algorithm Selection** (NEW - neither Eureka nor AgentHPO does this)
2. **Reward Function Design** (inspired by Eureka)
3. **Hyperparameter Tuning** (inspired by AgentHPO)

**Output**:
- Fully trained RL agent optimized for the task

**Target Environments**:
- Arkanoid (brick breaker game)
- Pingpong (pong game)
- Proly (physics-based control)

All adapted to Gymnasium API standard.

---

## 🔬 Background: AutoML and AutoRL

### AutoML (Automated Machine Learning)

**Definition**: End-to-end ML training tools that automate the entire pipeline.

**What AutoML Does**:
- Feature engineering
- Model selection
- Hyperparameter tuning
- Architecture search

**Example Tools**: Auto-sklearn, TPOT, H2O AutoML, Ray Tune

### AutoRL (Automated Reinforcement Learning)

**Definition**: Extension of AutoML principles to the RL domain.

**Why RL is Harder to Automate**:
- Noisy reward signals
- Sparse rewards
- Training instability
- Long training times
- Environment-specific challenges

**What AutoRL Should Automate**:
- Algorithm selection (Q-Learning, DQN, PPO, SAC, etc.)
- Hyperparameter tuning (learning rate, gamma, epsilon decay, etc.)
- Reward function design (shaping, scaling, components)
- Neural architecture design (layers, units, activations)

---

## 📚 Related Work: The Three Pillars

### 1. Eureka: Automated Reward Design

**Source**: NVIDIA Research - https://eureka-research.github.io/

**What It Does**:
- Reads environment source code
- Takes task description in natural language
- Generates reward function in Python using LLM (GPT-4)
- Uses evolutionary iteration with "Reward Reflection"

**Key Innovation: Reward Reflection Loop**
```
1. Generate population of reward functions (e.g., 16 candidates)
2. Train agents in parallel using each reward
3. Collect performance metrics + scalar breakdown
4. LLM analyzes WHY each reward failed/succeeded
5. Generate improved reward functions based on semantic feedback
6. Repeat until convergence
```

**Example**:
- Task: "Make robot hand spin a pen"
- Eureka generates reward with terms for:
  - Angular velocity of pen
  - Grip stability
  - Orientation alignment
  - Energy penalty

**Results**:
- 83% of tasks: Eureka rewards > human-designed rewards
- Achieved superhuman performance on dexterous manipulation

**Limitation for Us**:
- Runs on NVIDIA Isaac Gym (requires specific GPU setup)
- Long training time per iteration
- High compute cost

### 2. AgentHPO: Semantic Hyperparameter Optimization

**Source**: https://arxiv.org/abs/2402.01881

**What It Does**:
- Automates hyperparameter tuning using LLM reasoning
- Goes beyond numerical optimization (Bayesian, Grid Search)

**Key Innovation: Creator-Executor Architecture**

**Creator Agent** (Principal Investigator):
- Proposes initial hyperparameters based on dataset/model understanding
- Uses domain knowledge ("CNNs typically need learning rate in [1e-4, 1e-2]")

**Executor Agent** (Diagnostician):
- Runs experiment
- Monitors training logs in REAL-TIME
- Parses loss curves, gradient norms, etc.
- Provides SEMANTIC feedback:
  - ❌ Not: "Trial failed, score = -∞"
  - ✅ Instead: "Loss diverged after epoch 5 due to gradient explosion. Learning rate likely too high."

**Feedback Loop**:
```
Creator proposes → Executor trains → Executor diagnoses →
Creator reasons about failure → Creator proposes fix → Repeat
```

**Example Reasoning**:
```
Observation: Loss curve oscillated then NaN
Diagnosis: Gradient explosion
Root Cause: Learning rate too high OR batch size too small
Action: Decrease learning rate from 0.01 to 0.001 AND increase batch size
```

**Why It's Better Than Traditional HPO**:
- Sample efficient (fewer trials needed)
- Interpretable (explains WHY parameters were chosen)
- Human-like reasoning (mimics ML engineer workflow)

### 3. Agent²: End-to-End Multi-Agent System

**Source**: https://arxiv.org/html/2509.13368v1

**What It Claims**:
- Fully automated end-to-end AutoRL
- Multi-agent architecture with specialized roles
- Each agent handles different aspect of RL pipeline

**Our Knowledge Gap**:
- **Not open source** - cannot verify implementation
- Very complex architecture
- Unclear if it actually works as claimed

**Why We're Not Using It**:
- Cannot reproduce
- Possibly over-engineered
- Better to build from proven components (Eureka + AgentHPO)

---

## 💡 Our Innovation: The Missing Piece

### What Eureka Does:
✅ Reward function design

### What AgentHPO Does:
✅ Hyperparameter tuning

### What NEITHER Does:
❌ **Algorithm Selection**

### Our Contribution: Algorithm Selection

**The Problem**:
- Different RL algorithms suit different tasks:
  - **Discrete actions**: DQN, Q-Learning
  - **Continuous control**: PPO, SAC, TD3
  - **High-dimensional**: A3C, PPO
  - **Sample efficiency**: SAC, DDPG
  - **Stability**: PPO, TRPO

**Manual Selection is Hard**:
- Requires RL expertise
- Trial and error
- No systematic approach

**Our Automated Selection Approach**:

Given environment metadata, LLM selects algorithm based on:

```python
def select_algorithm(env_description, constraints):
    # Analyze action space
    if action_space == "discrete":
        candidates = ["DQN", "Q-Learning", "PPO"]
    else:
        candidates = ["PPO", "SAC", "TD3"]

    # Analyze observation space
    if observation_dim > 100:
        # High-dimensional -> need neural networks
        candidates.remove("Q-Learning")

    # Hardware constraints
    if memory_limited:
        # PPO requires critic network (doubles memory)
        candidates.remove("PPO")
        candidates.append("GRPO")  # More efficient

    # Sample efficiency requirements
    if sample_limited:
        prefer = ["SAC", "DDPG"]  # Off-policy, uses replay buffer

    # Stability requirements
    if stability_critical:
        prefer = ["PPO", "TRPO"]  # Clipped updates

    return llm_reason_and_select(candidates, task_description)
```

**LLM Reasoning Example**:
```
Task: "Train agent to play Arkanoid"
Analysis:
- Action space: Discrete (left, right, stay) → DQN or Q-Learning
- Observation space: [ball_x, ball_y, vx, vy, paddle_x] → Low-dimensional
- Hardware: Standard laptop → No restrictions
- Training budget: Moderate → Off-policy preferred (reuse samples)

Selection: DQN
Reasoning: Discrete actions make DQN ideal. Low observation dimensionality
means small network is sufficient. DQN's experience replay enables sample
efficiency within moderate training budget.
```

---

## 🏗️ System Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  USER INPUT                                                 │
│  - Environment code (Gymnasium-compatible)                  │
│  - Task description: "Train agent to play Arkanoid"         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: ALGORITHM SELECTION (LLM)                         │
│  - Parse env.action_space, env.observation_space            │
│  - Analyze task requirements                                │
│  - Select: Q-Learning / DQN / PPO / SAC / etc.              │
│  Output: {"algorithm": "DQN", "reasoning": "..."}           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: REWARD FUNCTION DESIGN (Eureka-inspired)          │
│  - Generate reward function code                            │
│  - Test population of reward variants                       │
│  - Reflection: Analyze which rewards work better            │
│  - Iterate and refine                                       │
│  Output: reward_function.py                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: HYPERPARAMETER TUNING (AgentHPO-inspired)         │
│  Creator: Propose initial hyperparameters                   │
│  Executor: Train with proposed params                       │
│  Executor: Monitor logs, diagnose issues                    │
│  Creator: Reason about failures, refine params              │
│  Iterate until convergence                                  │
│  Output: {lr: 0.001, gamma: 0.99, epsilon_decay: 0.995, ...}│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  TRAINING & EVALUATION                                      │
│  - Train agent with selected algorithm + reward + params    │
│  - Log metrics (TensorBoard)                                │
│  - Evaluate performance                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT                                                     │
│  - Trained model weights                                    │
│  - Performance metrics                                      │
│  - Complete configuration (algorithm + reward + params)     │
└─────────────────────────────────────────────────────────────┘
```

### Key Interfaces

1. **Gymnasium Environment Interface**
   - Standardized: `reset()`, `step()`, `action_space`, `observation_space`
   - LLM can parse these to understand environment

2. **Algorithm Selection Interface**
   ```json
   {
     "action_space": {"type": "discrete", "n": 3},
     "observation_space": {"type": "box", "shape": [5]},
     "task": "Hit all bricks without missing ball"
   }
   → LLM → {"algorithm": "DQN", "rationale": "..."}
   ```

3. **Reward Function Interface**
   ```python
   def reward(state, action, next_state, done):
       # LLM-generated code
       reward = 0
       if brick_hit:
           reward += 1
       if ball_missed:
           reward -= 1
       return reward
   ```

4. **Hyperparameter Interface**
   ```json
   {
     "learning_rate": 0.001,
     "gamma": 0.99,
     "epsilon_start": 1.0,
     "epsilon_decay": 0.995,
     "replay_buffer_size": 10000,
     "batch_size": 64
   }
   ```

5. **Training Metrics Interface**
   ```json
   {
     "episode": 100,
     "mean_reward": 15.3,
     "std_reward": 8.2,
     "convergence": false,
     "issue": "High variance, slow convergence"
   }
   → LLM analyzes → "Decrease learning rate, increase batch size"
   ```

---

## 📊 Benchmarking Strategy

### The Challenge

Most AutoRL papers benchmark against:
- Human expert designs
- Other AutoRL methods
- Random search

**Our Problem**: We don't have "human expert" reward functions or hyperparameters for our custom environments.

### Our Solution (Suggested by Gemini)

**Baseline Comparison**:

1. **Default Configuration (Baseline)**
   - Algorithm: Standard DQN
   - Reward: Simple sparse reward (e.g., +1 for winning, 0 otherwise)
   - Hyperparameters: Gymnasium/Stable-Baselines3 defaults

2. **Our LLM-Tuned Configuration**
   - Algorithm: LLM-selected
   - Reward: LLM-generated (dense, shaped)
   - Hyperparameters: LLM-optimized

**Evaluation Metrics**:
- **Convergence Speed**: Episodes to reach target performance
- **Final Performance**: Average reward over last 100 episodes
- **Stability**: Variance of rewards during training
- **Sample Efficiency**: Performance vs. number of environment interactions

**Experimental Setup**:
```
For each environment (Arkanoid, Pingpong, Proly):
  Run 1: Default (sparse reward + default hyperparams)
  Run 2: LLM-RL-Tuner (full pipeline)

  Compare:
  - Training curves
  - Convergence time
  - Final success rate
  - Reward variance

  Repeat with 5 random seeds for statistical significance
```

### Expected Results

**Hypothesis**:
- LLM-tuned should converge faster (fewer episodes)
- LLM-tuned should be more stable (lower variance)
- LLM-tuned should achieve higher final performance

**Validation**:
- If LLM-tuned is better → Our method works
- If comparable → Still valuable (automated vs. manual effort)
- If worse → Need to debug and understand failure modes

---

## 🛠️ Technical Implementation Plan

### Technology Stack

**RL Framework**:
- PyTorch (deep learning)
- Stable-Baselines3 (RL algorithms: DQN, PPO, SAC)
- Gymnasium (environment interface)

**LLM Integration**:
- Anthropic Claude 4.5 API (primary reasoning engine)
- OpenAI GPT-4 (backup/comparison)

**Environments**:
- Custom Gymnasium-compatible implementations:
  - Arkanoid (adapted from open source)
  - Pingpong (adapted)
  - Proly (adapted)

**Logging & Visualization**:
- TensorBoard (training curves)
- Weights & Biases (experiment tracking)
- Custom logging for LLM reasoning traces

**Compute**:
- Local: NVIDIA GTX 1080 Ti (11 GB VRAM)
- Alternative: Google Colab / Kaggle (for experiments)

### Implementation Phases

**Phase 1: Environment Setup** (Week 1)
- [ ] Adapt Arkanoid to Gymnasium API
- [ ] Adapt Pingpong to Gymnasium API
- [ ] Adapt Proly to Gymnasium API
- [ ] Verify all environments work with Stable-Baselines3
- [ ] Create baseline DQN training scripts

**Phase 2: Algorithm Selection** (Week 2)
- [ ] Implement env metadata extraction
- [ ] Design LLM prompt for algorithm selection
- [ ] Implement algorithm selection logic
- [ ] Test selection on all 3 environments
- [ ] Validate selections make sense

**Phase 3: Reward Function Design** (Week 3)
- [ ] Implement Eureka-style reward generation
- [ ] Create reward reflection loop
- [ ] Test reward functions in actual training
- [ ] Compare LLM-generated vs. sparse baseline

**Phase 4: Hyperparameter Tuning** (Week 4)
- [ ] Implement Creator-Executor architecture
- [ ] Create training log parser
- [ ] Implement semantic feedback loop
- [ ] Test on all 3 environments

**Phase 5: Integration & Testing** (Week 5)
- [ ] Integrate all three stages into end-to-end pipeline
- [ ] Run full experiments with 5 seeds each
- [ ] Collect all metrics
- [ ] Generate comparison graphs

**Phase 6: Paper Writing** (Week 6)
- [ ] Write Results section with graphs
- [ ] Write Discussion/Analysis
- [ ] Create final PDF

---

## 📖 Key Concepts & Definitions

### RL Fundamentals

**MDP (Markov Decision Process)**:
- State space S
- Action space A
- Transition dynamics P(s'|s,a)
- Reward function R(s,a,s')
- Discount factor γ

**Q-Learning**:
- Learn Q(s,a) = expected return from state s taking action a
- Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

**DQN (Deep Q-Network)**:
- Use neural network to approximate Q(s,a)
- Experience replay buffer
- Target network for stability

**PPO (Proximal Policy Optimization)**:
- Policy gradient method
- Clips policy updates to prevent large changes
- Requires critic (value function) network

### AutoRL Concepts

**Reward Shaping**:
- Adding intermediate rewards to guide learning
- Dense vs. sparse rewards
- Potential-based shaping (F(s) to avoid reward hacking)

**Hyperparameter Tuning**:
- Learning rate α: step size for updates
- Discount factor γ: future reward weighting
- Exploration: epsilon-greedy, entropy bonus
- Network architecture: layers, units, activations

**Algorithm Selection**:
- On-policy vs. off-policy
- Model-free vs. model-based
- Value-based vs. policy-based
- Discrete vs. continuous action spaces

---

## 🔗 References & Resources

### Papers

1. **Eureka** - "Eureka: Human-Level Reward Design via Coding Large Language Models"
   - Link: https://eureka-research.github.io/
   - Key: Reward Reflection loop, evolutionary generation

2. **AgentHPO** - "Agent-Driven Hyperparameter Optimization"
   - Link: https://arxiv.org/abs/2402.01881
   - Key: Creator-Executor architecture, semantic log analysis

3. **Agent²** - "Agent-Generates-Agent: Automated Agent Generation"
   - Link: https://arxiv.org/html/2509.13368v1
   - Key: Multi-agent coordination (but not open source)

4. **OPRO** - "Optimization by PROmpting"
   - Google DeepMind
   - Key: LLM as optimizer using prompt history

5. **DrEureka** - "Domain Randomization via LLM"
   - Extension of Eureka for Sim2Real transfer

### Tools & Libraries

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **Ray Tune**: https://docs.ray.io/en/latest/tune/
- **Optuna**: https://optuna.org/ (Bayesian optimization)

### Gemini Conversation

- Full analysis: https://gemini.google.com/share/21c3881d9e9f
- Key insights on benchmarking strategy
- Comprehensive AutoRL landscape overview

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)

1. ✅ All 3 environments working with Gymnasium API
2. ✅ Algorithm selection produces sensible choices
3. ✅ Reward functions are syntactically valid Python
4. ✅ Hyperparameter tuning completes without crashes
5. ✅ End-to-end pipeline runs without manual intervention

### Research Success

1. 📈 LLM-tuned converges faster than baseline (statistical significance)
2. 📊 LLM-tuned achieves higher final performance
3. 📉 LLM-tuned has lower training variance
4. 📝 All reasoning traces are interpretable and logical

### Publication Quality

1. 📄 Clear methodology description
2. 📊 Comprehensive experimental results
3. 🔬 Ablation studies (what if we remove algorithm selection?)
4. 💡 Novel insights about LLM reasoning in AutoRL
5. 🔓 Open source release of code

---

## ⚠️ Known Challenges & Risks

### Technical Risks

1. **LLM Hallucination**
   - Risk: Generated reward code has bugs
   - Mitigation: Syntax validation, test execution, fallback to baseline

2. **Sample Inefficiency**
   - Risk: Each training run takes hours
   - Mitigation: Use short training runs for tuning, final run for evaluation

3. **Non-Determinism**
   - Risk: Same prompt → different outputs
   - Mitigation: Multiple seeds, report mean and variance

4. **Token Costs**
   - Risk: High API costs for Claude 4.5
   - Mitigation: Use caching, optimize prompts, consider local models

### Research Risks

1. **Baseline Too Weak**
   - Risk: Easy to beat bad baseline
   - Mitigation: Also compare against human-tuned configuration

2. **Overfitting to Environments**
   - Risk: Method only works on our 3 games
   - Mitigation: Test on additional Gym environments (CartPole, MountainCar)

3. **No Improvement**
   - Risk: LLM tuning doesn't help
   - Response: Still valuable negative result, analyze why

---

## 📅 Timeline

**Total Duration**: 6 weeks

- **Week 1**: Environment setup
- **Week 2**: Algorithm selection
- **Week 3**: Reward design
- **Week 4**: Hyperparameter tuning
- **Week 5**: Integration & experiments
- **Week 6**: Paper writing & submission

**Deadline**: [Insert conference deadline here]

---

## 👥 Team & Responsibilities

- **張昀棠 (F74111160)**: [Role TBD]
- **張羿軒 (F74114760)**: [Role TBD]
- **劉力瑋 (F74114728)**: [Role TBD]
- **劉柏均 (F74112297)**: [Role TBD]

---

*Last Updated: 2024-11-27*
*Document Version: 1.0*
