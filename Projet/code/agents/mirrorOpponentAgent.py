class MirrorOpponentAgent:

    def __init__(self):
        pass

    def act(self,info):
        if info[0].observation.step > 0:
            return info[0].observation.lastOpponentAction
        else:
            return  int(np.random.choice(2,1)[0])

    def play(self, obs, configuration):
        if obs.step > 0:
            return obs.lastOpponentAction
        else:
            return 0