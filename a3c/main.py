
from agent import A3C


def main():
    """
        A3C runner
    """
    state_size = 1120
    action_size = 987 + 21  # topological actions + redispatch

    agent = A3C(state_size, action_size)
    agent.load_checkpoint()

    agent.train_workers()


if __name__ == '__main__':
    main()
