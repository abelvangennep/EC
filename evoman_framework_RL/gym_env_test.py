import cv2
import sys
# import threading
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

sys.path.insert(0, 'evoman')
from gym_environment import Evoman

# class envTrain (threading.Thread):
#     def __init__(self, model, timesteps):
#         self.model = model
#         self.timesteps = timesteps
#         threading.Thread.__init__(self)
#     def run(self):
#         self.model.learn(self.timesteps)


environments = [MaxAndSkipEnv(Evoman(enemyn=sys.argv[1]), skip=2)]

model = PPO('MlpPolicy', environments[0], verbose=sys.argv[1]=='1')

for env in environments:
    model.set_env(env)
    model.learn(total_timesteps=(2 ** 17))

    print(f'\n\n\nFinished learning env{sys.argv[1]}!\n\n\n')

    # env.env.keep_frames = True
    # obs = env.reset()
    #
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fps = 30
    # video_filename = f'env{i}_({fsn}, 10,2).avi'
    # out = cv2.VideoWriter(video_filename, fourcc, fps, (env.WIDTH, env.HEIGHT))
    # for _ in range(2500):
    #     action, _state = model.predict(obs, deterministic=False)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         for frame in env.render('video'):
    #             out.write(frame)
    #         break
    # out.release()
