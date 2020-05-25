from hyperparameters import *
from Agent import *
from CNN import *
from helper import *
import plot

def test(env,agent):
    run = 'y'
    while run == 'y' or run == 'Y':
        obsv = process_state(env.reset())

        if agent.game=='Breakout-v4' or agent.game=='Breakout-v0':
            # Fire to start the game
            env.step(1)
        
        current_state = np.array([obsv, obsv, obsv, obsv])
        current_lives = TOTAL_LIVES  # Total Number of lives
        done = False
        
        # for _ in range(randint(1, NOOPMAX)):
        #     obsv, _, _, _ = env.step(0)
        #     obsv = process_state(obsv)
        #     next_state = get_next_state(current_state, obsv)
        #     current_state = next_state

        score = steps = reward = 0

        while not done:
            steps += 1
            env.render()
            sleep(0.01)
            action = agent.action(np.asarray([current_state]))

            obsv, reward, done, info = env.step(action)
            obsv = process_state(obsv)
            next_state = get_next_state(current_state, obsv)

            current_state = next_state
            
            if agent.game=='Breakout-v4' or agent.game=='Breakout-v0':
                if info['ale.lives'] < current_lives:
                    env.step(1) # Fire to continue game
                    current_lives = info['ale.lives']
                    # print('Lives Left = ', current_lives, '\tGame Over = ', done)

            if agent.game=='Pong-v4' or agent.game=='Pong-v0':
                if reward==1:
                    env.step(1) # Fire to take service

            score += reward
            
        print("Total Reward: ", score, "\nSteps: ", steps)
        run = input("\nRUN TEST AGAIN? (Y/N) : ")
    print("Exiting Environment.")

def initial_exploration(env,agent):

    log = open("log_"+game+".txt", 'w')
    log.write("\n\n=================================  Starting Exploration  ==============================\n")
    print('Begining to Explore!!')
    
    for episode in range(NUM_EXPLORE):
        timer = time()
        done = False
        current_lives = TOTAL_LIVES  # Total Number of lives
        avg_score = 0

        obsv = process_state(env.reset())

        if agent.game=='Breakout-v4' or agent.game=='Breakout-v0':
            # Fire to start the game
            env.step(1)
            
        current_state = np.array([obsv, obsv, obsv, obsv])

        for _ in range(randint(1,NOOPMAX)):
            obsv, _, _, _ = env.step(0)
            obsv = process_state(obsv)
            next_state = get_next_state(current_state, obsv)
            current_state = next_state

        steps = 0
        score = 0
        
        while not done:
            steps += 1

            action = randint(0,K-1)

            obsv, reward, done, info = env.step(action)
            obsv = process_state(obsv)

            next_state = get_next_state(current_state, obsv)

            clipped_reward = np.clip(reward, -1, 1)
            agent.experience_gain(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

            current_state = next_state
            
            if agent.game=='Breakout-v4' or agent.game=='Breakout-v0':
                if info['ale.lives'] < current_lives:
                    env.step(1) # Fire to continue game
                    current_lives = info['ale.lives']
                    # print('Lives Left = ', current_lives, '\tGame Over = ', done)

            if agent.game=='Pong-v4' or agent.game=='Pong-v0':
                if reward==1:
                    env.step(1) # Fire to take service
                
            score += reward
        
        timer = time() - timer
        avg_score = (avg_score + score)/2 if episode != 0 else score
        log.write(str(episode+1) + "\tTotalReward = " + str(score) + "\tSteps: " + str(steps) + "\tMoving Avg: {:.2f}".format(avg_score) + "\tTime: %d" % int(timer/60) + ":{:.0f} \n".format((timer % 60)))
        print(episode+1, "\tTotalReward = ", score, "\tSteps: ", steps, "\tMoving Avg: {:.2f}".format(avg_score),"\tTime: %d" % (timer/60), "\b:{:.0f}".format((timer % 60)))
    
    log.write("=================================   Ending Exploration   ==============================\n")
    log.close()
    agent.save_state()
    print("\nEXPLORATION COMPLETED\n")

def train(env,agent):
    log = open("log_"+game+".txt", 'a')
    log.write("\n\n=================================  Starting Training Session  ==============================\n")
    timer = time()
    no_episodes = int(input("Number of Episodes? : "))
    
    total_steps = 0
    avg_score = 0

    for episode in range(no_episodes):
        timer = time()
        done = False
        current_lives = TOTAL_LIVES  # Total Number of lives
        
        if agent.game=='Breakout-v4' or agent.game=='Breakout-v0':
            # Fire to start the game
            env.step(1)
  
        obsv = process_state(env.reset())
        current_state = np.array([obsv, obsv, obsv, obsv])
        
        for _ in range(randint(1,NOOPMAX)):
            obsv,_,_,_ = env.step(0)
            obsv = process_state(obsv)
            next_state = get_next_state(current_state, obsv)
            current_state = next_state
        
        steps = 0
        score = 0
        
        while not done:
            steps+=1
            total_steps+=1

            action = agent.action(np.asarray([current_state]))
            agent.greedy()
            
            obsv, reward, done, info = env.step(action)
            obsv = process_state(obsv)

            next_state = get_next_state(current_state, obsv)

            clipped_reward = np.clip(reward, -1, 1)
            agent.experience_gain(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

            if steps%4==0:
                 agent.train()
                    
            if total_steps%TARGET_UPDATE_PERIOD==0:
                agent.update_target_network()
    
            current_state = next_state

            if agent.game=='Breakout-v4' or agent.game=='Breakout-v0':
                if info['ale.lives'] < current_lives:
                    env.step(1) # Fire to continue game
                    current_lives = info['ale.lives']
                    # print('Lives Left = ', current_lives, '\tGame Over = ', done)

            if agent.game=='Pong-v4' or agent.game=='Pong-v0':
                if reward==1:
                    env.step(1) # Fire to take service
        
            score += reward
        
        timer = time() - timer
        avg_score = (avg_score + score)/2 if episode != 0 else score
        log.write(str(episode+1) + "\tTotalReward = " + str(score) + "\tSteps: " + str(steps) + "\tMoving Avg: {:.2f}".format(avg_score) + "\tTime: %d" % int(timer/60) + ":{:02.0f} \n".format((timer % 60)))
        print(episode+1, "\tTotalReward = ", score, "\tSteps: ", steps, "\tMoving Avg: {:.2f}".format(avg_score),"\t Current Episilon = %f "%(agent.epsilon),"\tTime: %d" % (timer/60), "\b:{:.0f}".format((timer % 60)))
        agent.save()
    
    log.write("=================================   Ending Training Session   ==============================\n")
    log.close()
    print("\nTraining Completed\n")

    agent.save_state()

    del timer
    del current_state
    del current_lives
    del next_state
    del clipped_reward
    del score
    del avg_score
    del steps
    del log
    del obsv
    del env
    gc.collect()

if __name__ == "__main__":
    print('Train or Test any atari games on this DDQN Network\n')
    # Note : For games like breakout, where manual fire is necessary only a part of code in Training and Exploration needs to be changed.
    try:
        game_name = sys.argv[1]
    except IndexError:
        game_name = "Breakout"

    game = game_name + "-v4"

    while(1):
        choice = input("\n1. Train Agent\n2. Run Test\n3. View results of pre-trained weights\n\n")
        if  choice == '1':
            _log = input('Log Model History? (y/n):')
            _log = True if _log=='y' else False if _log=='n' else None
            env = gym.make(game)
            agent = Agent((84, 84, 4),K, game, load_weights=True)
            if agent.load_state(True):
                train(env,agent, _log)
            else:
                initial_exploration(env,agent)
                train(env,agent, _log)
            env.close()
            break
        elif choice == '2':
            env = gym.make(game)
            agent = Agent((84, 84, 4),K, game, load_weights=True)
            test(env,agent)
            env.close()
            break
        elif choice == '3':
            # Pass graph generation function open using matplotlib
            plot.main("log_"+game_name+".txt")
            break
        else :
            print('Please Enter a valid choice\n')
