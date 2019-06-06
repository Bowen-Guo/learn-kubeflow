import subprocess
import sys


def run(command: str, timeout=60000):
    if not command:
        return

    print(f">> {command}")
    return subprocess.Popen(command.split(), stdout=sys.stdout, stderr=sys.stderr).wait(timeout=timeout)


INITIAL_COMMANDS = '''
pwd
'''.splitlines()


if __name__ == '__main__':
    for command in INITIAL_COMMANDS:
        run(command)

    ret = run(' '.join(sys.argv[1:]))
    exit(ret)
