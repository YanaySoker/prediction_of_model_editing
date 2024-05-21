NUM_OF_LAYERS = 48
HIGH_RES_LAYERS_RANGE = range(0, NUM_OF_LAYERS)

train_set = ['The native language of {} is', 'In {}, an official language is', 'Where is {}? It is located in', '{} is headquartered in', '{} is to debut on', '{} holds a citizenship from', '{} is a product of', "{}'s capital is", '{}, who has a citizenship from', '{} was originally from', '{} is the capital city of', '{} was created by', '{}, a product manufactured by', '{} was developed in', 'The capital of {} is', '{}, founded in', 'In {}, the language spoken is', "{}'s headquarters are in", '{} passed away in', '{}, which is located in', '{} is in', '{} is a twin city of', '{} debuted on', '{}, a citizen of', 'The mother tongue of {} is', '{} can be found in', 'What sport does {} play? They play', '{} is based in', '{}, located in', '{}, developed by', '{}, produced by', '{} is a native speaker of', '{} was originally aired on', '{} is developed by', '{}, that originated in', '{}, which has the capital', '{}, who is employed by', '{} is located in the country of', 'The capital city of {} is', '{}, a product developed by', 'The headquarter of {} is located in', '{} is affiliated with the', '{}, whose headquarters are in', '{}, a product created by', '{}, created by', '{} is originally from', '{} is a part of the continent of', '{} is produced by', '{} has a citizenship from', '{} was born in', '{} is owned by', '{} is created by', '{} originated in', 'The official religion of {} is', '{} is a professional', '{}, a product of']
train_sets = [[train_set[i] for i in range(len(train_set)) if i%4==j] for j in range(4)]

test_set = ["{}'s capital,", '{} died in the city of', '{} is native to', '{} belongs to the continent of', 'The official language of {} is', '{} works in the field of', '{}, who is a citizen of', '{}, who holds a citizenship from', '{} was a product of', '{}, in', '{} plays in the position of', "{}'s capital city,", '{} premiered on', '{} is located in', '{} worked in the city of', '{} premieres on', '{} was developed by', '{} was native to', '{} is a citizen of']
# 17, 18, 30, 31, 32, 33, 34, 35, 45, 46, 47, 48, 49, 52, 63, 64, 66, 68, 70, 75, 76, 79, 80, 86, 87, 88, 89, 93, 94, 95, 96, 128, 129, 130, 131,

SET_NAMES = {"all": train_set + test_set, "train": train_set, "train0": train_sets[0], "train1": train_sets[1], "train2": train_sets[2], "train3": train_sets[3], "test": test_set}

class nan:
	pass

class inf:
	pass

# fixing bad values obtained from division by 0
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


# settings for FLS, OLU, probs analysis:

k = "prob"    # int or str from ["prob", "ratio prob", "increase v", "increase l"]
min_n = 5
n_values = 30   # number of values to discreet continuous values (prob, ratio prob)
r_set = train_set+test_set
p_threshold = 0.0

