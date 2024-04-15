import matplotlib.pyplot as plt
import os


def save_log(fed_name, accs, type):
    file = open("plots/" + type + '/' + fed_name + '.txt', 'w')
    for acc in accs:
        file.write(str(acc) + "\n")
    file.close()


def plot_acc(dict_val, type):
    plt.figure(figsize=(5, 4))
    num_rounds = 0
    for key in dict_val.keys():
        plt_vals = dict_val[key]
        plt.plot(list(range(1, len(plt_vals) + 1)), plt_vals, label=key)
        num_rounds = len(plt_vals)
    plt.legend()
    plt.xlabel('# of Rounds')
    plt.ylabel(f'Test {type}')
    plt.title(f'{type} vs Rounds')
    plt.xlim(0, num_rounds)
    plt.savefig(f'C:/Users/noroo/PycharmProjects/FederateLearning/plots/{type}')


def read_file(file_name, weight):
    data_pts = []
    with open(file_name, 'r') as f:
        for i in f.readlines():
            data_pts.append(float(i) * weight)
    return data_pts


if __name__ == "__main__":
    data_path = "C:/Users/noroo/PycharmProjects/FederateLearning/plots/"
    data_dict = {}
    for type in ['acc', 'loss']:
        for file in os.listdir(data_path + type):
            if os.path.join(data_path, type, file).endswith(".txt"):
                data = read_file(os.path.join(data_path, type, file), weight=1 if type == 'loss' else 100)
                if 'fed_avg' in file:
                    data_dict['FedAVG'] = data
                if 'baseline' in file:
                    data_dict['BaseLine'] = data
                elif 'fed_prox' in file:
                    data_dict['FedProx'] = data
                elif 'cwt' in file:
                    data_dict['cwt'] = data
                elif 'scaffold' in file:
                    data_dict['SCAFFOLD'] = data
                elif 'ditto' in file:
                    data_dict['DITTO'] = data
                elif 'pw' in file:
                    data_dict['PW'] = data

        plot_acc(data_dict, type)
