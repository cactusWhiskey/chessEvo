from tf_agents.metrics import py_metrics
import ChessAgent
import ChessEnv

collect_steps_per_iteration = 1
train_iterations = 10

# Metrics
train_metrics = [
    py_metrics.NumberOfEpisodes(),
    py_metrics.EnvironmentSteps(),
    py_metrics.AverageReturnMetric(),
    py_metrics.AverageEpisodeLengthMetric(),
]

env = ChessEnv.ChessEnv()

agent = ChessAgent.ChessAgent(env.time_step_spec(), env.action_spec())
agent.collect_initial_data(env, max_episodes=7000)
agent.train(env, train_iterations, collect_steps_per_iteration, train_metrics, play_test_every=200)
trans = agent.test_play(env)
print(trans[-1][2].reward)

env.close_engine()
print("Avg Return: " + str(train_metrics[2].result()))