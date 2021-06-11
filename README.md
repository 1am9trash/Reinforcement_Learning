強化學習筆記
---

- **發起時間**：2021/06/04
- **背景介紹**：原本對強化學習的認知只有經典的馬可夫決策過程（MDP），但逐漸發現這樣基礎的模型有許多能力與實現上的限制，於是開始學習其他的強化學習算法，並藉此紀錄過程。

- **更新**：
  | 更新日期    | 算法   |
  | ---        | ---   |
  | 2021/06/10 | DQN及相關優化算法 |

1. DQN及相關優化算法：
   - 筆記：[算法分析與實現成果](http://htmlpreview.github.io?https://github.com/1am9trash/Reinforcement_Learning/blob/main/dqn_models/dqn_models_note.html)
   - 簡介：分別使用DQN、nature DQN、DDQN等模型，在CartPole（台車木棒平衡遊戲）的環境裡學習，並達成遊戲目標。
   - 由模型操控遊戲的Demo：

     <video width="400" height="280" controls>
       <source src="./dqn/src/demo.mov">
	 </video>