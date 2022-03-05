from ML.ChessAgent import ChessAgent
from ML.ChessEnv import ChessEnv

env = ChessEnv()
agent = ChessAgent(env.time_step_spec(), env.action_spec())
agent.load_checkpoint()
agent.test_play(env, mode="supr")
env.close()