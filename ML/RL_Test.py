from ML import ChessEnv
import ChessAgent

train_iterations = 100000

env = ChessEnv.ChessEnv()

agent = ChessAgent.ChessAgent(env.time_step_spec(), env.action_spec())
agent.collect_initial_data(env, max_episodes=4000)
agent.train(env, train_iterations, play_test_every=200)
trans = agent.test_play(env)
print(trans[-1][2].reward)

env.close_engine()
print("Avg Return: " + str(agent.train_metrics[2].result()))
