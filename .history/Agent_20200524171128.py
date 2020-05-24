from hyperparameters import *
from CNN import *
from helper import *

class Agent:
    def __init__(self, input_shape, action_space, game_name,
                 memory=MAX_EXPERIENCES,
                 epsilon=1.0,
                 min_epsilon=MIN_EPSILON,
                 decay_rate=0.9999,
                 batch_size=BATCH_SIZE,
                 load_weights=True,
                 test=False):
        self.action_set = action_space  # if action_space <= 6 else 6
        self.input_shape = input_shape
        self.memory_size = memory
        self.epsilon = epsilon
        self.epsilon_min = min_epsilon
        self.decay = decay_rate
        self.batch_size = batch_size
        self.game = game_name
        print("Action Set: ", self.action_set)
        filepath = str("model.h5") if load_weights else None
        # loaded_model.load_weights("model.h5") if load_weights else None

        self.policy_network = CNN(self.input_shape,
                                  self.action_set,
                                  batch_size=self.batch_size,
                                  weights=filepath if os.path.exists(filepath) else None)
        if not test:
            self.target_network = CNN(self.input_shape, self.action_set, batch_size=self.batch_size)

            self.target_network.model.set_weights(self.policy_network.model.get_weights())

            self.experiences = []

    def action(self, state):
        if random() < self.epsilon:
            return randint(1, self.action_set)
        else:
            return self.policy_network.predict(state).argmax()+1

    def move(self, state):
        if random() < EXPLORATION_TEST:
            return randint(1, self.action_set)
        else:
            return self.policy_network.predict(state).argmax()+1

    def experience_gain(self, current_state, action, reward, next_state, done):
        if self.experiences.__len__() >= self.memory_size:
            self.experiences.pop(0)

        self.experiences.append({'current': current_state,
                                 'action': action,
                                 'reward': reward,
                                 'next_state': next_state,
                                 'done': done})

    def sample_experiences(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.experiences[randrange(0, self.experiences.__len__())])
        return np.asarray(batch)

    def train(self):
        self.policy_network.train(self.sample_experiences(), self.target_network)

    def greedy(self):
        if self.epsilon * self.decay > self.epsilon_min:
            self.epsilon *= self.decay
        else:
            self.epsilon = self.epsilon_min
            
#         print('Episilon Value ',self.episilon)

    def update_target_network(self):
        self.target_network.model.set_weights(self.policy_network.model.get_weights())
        print("Target Network Updated.")

    def experience_available(self):
        return True if self.experiences.__len__() >= self.batch_size else False

    def save(self):
        self.policy_network.save("model.h5")
        # self.policy_network.save(filepath=str(self.game+"_weights"))

    def save_state(self):
        
        while len(self.experiences)>MAX_EXPERIENCES:
          self.experiences.pop(0)

        print("Saving Experience State...")
        # gc.disable()
        with open("Experiences"+self.game, 'wb') as experience_dump:
            print('Episilon Value Saving : ',self.epsilon)
            joblib.dump((self.experiences,self.epsilon), experience_dump, compress=3)   
            # p.dump((self.experiences,self.epsilon), experience_dump, protocol=p.HIGHEST_PROTOCOL)
        # gc.enable()
        experience_dump.close()
        print("Number of Experiences : ", self.experiences.__len__())
        print("Experience State Saved!")

    def load_state(self, load=True):
        if load:
            if os.path.exists("Experiences"+self.game):
                print("Loading Experiences...")
                gc.disable()
                with open("Experiences"+self.game, 'rb') as experience_dump:
                    self.experiences,self.epsilon = joblib.load(experience_dump)
                gc.enable()
                experience_dump.close()
                gc.collect()
                print(self.experiences.__len__(), " EXPERIENCES LOADED SUCCESSFULLY\n\n\n\n")
                print("Current Episilon Value : ",self.epsilon)
                return True
        return False