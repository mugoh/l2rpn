from agent import A3C
import constants


def main():
    """
        A3C runner
    """
    constants.init()

    state_size = 1120
    action_size = 987 + 21  # topological actions + redispatch
    # action_size = constants.actions_array.shape[0]

    agent = A3C(state_size, action_size)
    # agent.load_checkpoint()

    agent.train_workers()


if __name__ == '__main__':
    main()
