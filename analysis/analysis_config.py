import matplotlib.pyplot as plt

NUM_OF_LAYERS = 48
HIGH_RES_LAYERS_RANGE = range(0, NUM_OF_LAYERS)

DEFAULT_OLU = {"finals": 4,
			   "probs": 8,
			   "plus": 4}

colors = ['aqua', 'aquamarine', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'hotpink', 'indianred', 'indigo', 'khaki', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'yellow', 'yellowgreen']


neighboring_by_probs = True
neighboring_by_true_probs = True
neighboring_by_finals = True
neighboring_by_plus = True


class nan:
	pass

class inf:
	pass


def change_to_none(X, print_count = True):
	count = 0
	for i in range(len(X)):
		if not (isinstance(X[i], float) or isinstance(X[i], list) or isinstance(X[i], dict) or isinstance(X[i], int) or X[i] == None):
			X[i]=None
			count+=1
		if isinstance(X[i], list):
			count+=change_to_none(X[i], False)
		if isinstance(X[i], dict):
			count+=change_to_none(list(X[i].values()), False)

	if print_count:
		print("Changes:", count)
	return count


def merge_res_dict(dict_low, dict_high):
	new_dict = {'neighborhood_scores': [], 'efficacy_scores': []}
	new_dict['neighborhood_scores'] += dict_low['neighborhood_scores']
	new_dict['neighborhood_scores'] += dict_high['neighborhood_scores']
	new_dict['efficacy_scores'] += dict_low['efficacy_scores']
	new_dict['efficacy_scores'] += dict_high['efficacy_scores']
	return new_dict


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

k = 1    # int or str from ["prob", "ratio prob", "increase v", "increase l"]
min_n = 5
n_values = 30   # number of values to discreet continuous values (prob, ratio prob)
p_threshold = 0.0

def print_list(list_input, file_name=None):
  file = open(file_name, "w", encoding="utf-8")
  file.write("d = [\n")
  for item in list_input:
    file.write(f"\t{item},\n")
  file.write("]")
  file.close()
