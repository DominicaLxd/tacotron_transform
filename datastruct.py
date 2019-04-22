from collections import deque
## deque 队列

def search(lines, pattern, history=5):
    previous_lines = deque(maxlen=history)
    for line in lines:
        if pattern in line:
            yield line, previous_lines
        previous_lines




if __name__ == '__main__':
    with open('./1.txt') as f:
        for line, prevlines in search(f, 'python', 5):
