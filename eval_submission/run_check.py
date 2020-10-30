import sys

from check_your_submission import main as test_submission


def check_submission():

    try:
        # model_dir = '../submission_1'
        model_dir = sys.argv[1]
    except IndexError:
        print(
            'Error: Missing submission dir\n\n[Usage: python3 run_check submission_dir]')

        sys.exit()

    else:
        test_submission(model_dir)


if __name__ == '__main__':
    check_submission()
