from random import randint


def random_shuffle(str):
    input_list = list(str)
    size = len(str)
    for i in range(size - 1, 0, -1):
        index_swap_to = i
        index_swap_from = randint(0, i - 1)
        swap_positions(input_list, index_swap_to, index_swap_from)
        return "".join(input_list)


def swap_positions(input_list, pos1, pos2):
    input_list[pos2], input_list[pos1] = input_list[pos1], input_list[pos2]
    return input_list


if __name__ == '__main__':
    JSON_KEY = "123456789"
    with open('test.json', 'w+') as fh:
        fh.write("{")
        for _ in range(1500000):
            key = random_shuffle(JSON_KEY)
            value = random_shuffle(JSON_KEY)
            line_content = '"{}":{},'.format(key, value)
            fh.write(line_content)
        fh.write('"end": 1')
        fh.write("}")
