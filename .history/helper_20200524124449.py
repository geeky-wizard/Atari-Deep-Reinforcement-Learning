from pillow import Image

def process_state(observation):
    img = np.asanyarray(Image.fromarray(observation, 'RGB').convert('L').resize((84, 110)))
    img = np.delete(img, np.s_[-10:], 0) 
    img = np.delete(img, np.s_[:16], 0)  
    return img

def get_next_state(current, observation):
    return np.append(current[1:], [observation], axis=0)

def processed_screen():
    name = "Seaquest-v4"
    env = gym.make(name)
    env.reset()
    for i in range(400):
        env.step(env.action_space.sample())
    screen = env.render("rgb_array")
    Image.fromarray(screen).save(name+".png")
    Image.fromarray(process_state(screen)).show()