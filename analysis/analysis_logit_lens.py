from neighborhood import d as neighborhood
from analysis_config import *
import matplotlib.pyplot as plt
from scores_0_6__7_24__26_51__52_80__82_86__88_102 import d as high_res_results
from probs_in_last_layer_0_125 import d as prob_dict_and_labels
from labels_dict_0_125 import d as labels_dict
from analysis_config import *
import numpy as np
import math

LABELS_DICT = labels_dict
RESULTS_LIST = high_res_results
LAYERS_RANGE = HIGH_RES_LAYERS_RANGE

FLS_DICT = dict()
FINAL_PROBS = {"prob": None, "ratio": None}

def optimal_layer_to_update_one_dict(results_dict, olu_type="harmonic", parameter=1, efficacy_probs=True, return_performance=False):
    keys = ["finals", None, "true", None, "probs plus"]
    olu_dict = {key: [] for key in keys}
    scores = results_dict["neighborhood_scores"]

    afficacy_idx = int(efficacy_probs)
    efficacy = [results_dict["efficacy_scores"][l][afficacy_idx] for l in range(len(results_dict["efficacy_scores"]))]

    n_subjects = len(efficacy[0])

    for i in range(n_subjects):
            efficacy_i = [efficacy[l_idx][i] for l_idx in range(len(efficacy))]
            for k in range(len(keys)):
                key = keys[k]
                if key==None:
                    continue
                k_scores = [_list[k] for _list in scores]
                scores_of_subject_i = [k_scores[l_idx][i] for l_idx in range(len(k_scores))]

                if key=="finals":
                    scores_of_subject_i = [1-v for v in scores_of_subject_i]

                olu = new_olu(scores_of_subject_i, efficacy_i, LAYERS_RANGE, olu_type, parameter, return_performance)
                olu_dict[key].append(olu)

    return olu_dict  # {key: {subject: olu}}


def transpose_matrix(matrix):
    new_matrix = [[line[k] for line in matrix] for k in range(len(matrix[0]))]
    return new_matrix


def scores_of_relation(results_dict):
    keys = ["finals", "true", "probs plus"]
    success_dict = {key: [] for key in keys}

    orig_spcf_scores = results_dict["neighborhood_scores"]
    spcf_finals = [orig_spcf_scores[l][0] for l in range(len(LAYERS_RANGE))] # n_layers X n_subjects
    spcf_true = [orig_spcf_scores[l][2] for l in range(len(LAYERS_RANGE))]
    spcf_plos = [orig_spcf_scores[l][4] for l in range(len(LAYERS_RANGE))]
    spcf_scores = []
    for spcf_matrix in [spcf_finals, spcf_true, spcf_plos]:
        spcf_scores.append(transpose_matrix(spcf_matrix))

    efficacy_finals = [results_dict["efficacy_scores"][l][0] for l in range(len(results_dict["efficacy_scores"]))]
    efficacy_probs = [results_dict["efficacy_scores"][l][1] for l in range(len(results_dict["efficacy_scores"]))]
    efficacy_scores = []
    for eff_matrix in [efficacy_finals, efficacy_probs]:
        efficacy_scores.append(transpose_matrix(eff_matrix))

    n_subjects = len(efficacy_finals[0])

    for eff_idx, spcf_idx in [(0,0), (1,1), (1,2)]:
        current_aff = efficacy_scores[eff_idx]
        current_spcf = spcf_scores[spcf_idx]
        key_name = keys[spcf_idx]

        for i in range(n_subjects):
            i_eff = current_aff[i]
            i_spcf = current_spcf[i]
            success = [harmonic_mean(i_eff[j], i_spcf[j]) for j in range(len(LAYERS_RANGE))]

            success_dict[key_name].append(success)

    return success_dict  # {key: {subject: [success X n_layers]]}}


def new_olu(spcf_list, eff_list, layers, olu_type="harmonic", parameter=1.0, return_performance = False):
    f = {"harmonic": harmonic_mean, "mean": mean, "eff": take_second, "spcf": take_first}[olu_type]
    scores = [f(spcf_list[i], eff_list[i], parameter) for i in range(len(spcf_list))]
    if return_performance:
        return scores
    olu_idx = np.argmax(scores)
    olu = layers[olu_idx]
    return olu


def take_first(val1, val2, _=None):
    return val1


def take_second(val1, val2, _=None):
    return val2


def optimal_layer_to_update_all_dicts(olu_type,efficacy_threshold=1, efficacy_probs=True, return_performance = False):
    keys = ["finals", None, "true", None, "probs plus"]
    olu_dict = {key: {} for key in keys}
    for i in range(len(RESULTS_LIST)):
        res_dict = RESULTS_LIST[i]
        if res_dict==None:
            continue
        relation, subjects, _, _ = neighborhood[i]
        if relation not in olu_dict[keys[0]].keys():
            for key in keys:
                olu_dict[key][relation] = {}

        r_olus = optimal_layer_to_update_one_dict(res_dict, olu_type, efficacy_threshold, efficacy_probs, return_performance)

        for key in keys:
            if key==None:
                continue
            for s_idx in range(len(subjects)):
                subject = subjects[s_idx]
                try:
                    olu = r_olus[key][s_idx]
                except:
                    print("error", i, relation, "::", subject)
                olu_dict[key][relation][subject] = olu

    return olu_dict


def two_categories_dict(category1, category2):
    dict1 = {}
    dict2 = {}

    for i in range(len(RESULTS_LIST)):
        res_dict = RESULTS_LIST[i]
        if res_dict==None:
            continue
        relation, subjects, _, _ = neighborhood[i]
        if relation not in dict1.keys():
            dict1[relation] = []
            dict2[relation] = []

        success = scores_of_relation(res_dict)

        for s_idx in range(len(subjects)):
                subject = subjects[s_idx]
                try:
                    current_success = success["probs plus"][s_idx]
                except:
                    print("error", i, relation, "::", subject)

                if current_success[category1]>current_success[category2]:
                    dict1[relation].append(subject)
                else:
                    dict2[relation].append(subject)

    return dict1, dict2


def common_value(values_list):
    histogram = {}
    for v in values_list:
        if v not in histogram.keys():
            histogram[v]=0
        histogram[v]+=1
    return max(histogram.keys(), key=lambda x: histogram[x])


def first_label_success(labels, true_label, contin_flag=False):
    n_layers = len(labels)

    if contin_flag:
        for i in range(n_layers - 1, -1, -1):
            if true_label not in labels[i]:
                return i + 1
    else:
        for i in range(n_layers):
            if true_label in labels[i]:
                return i


def take_first_k(labels, k):
    # labels: [[...], [...], ...]
    new_labels = [l[:k] for l in labels]
    return new_labels


def fls_prob(k, ration_flag=False, min_n=0, mlp_flag = False):
    fls_dict = forward_success_dict(k, mlp_flag)
    p_dict = final_prob_dict(ratio=ration_flag)

    X = []  # FLS
    Y = []  # p

    for r in p_dict.keys():
        if r in fls_dict.keys():
            p_dict_r = p_dict[r]
            fls_dict_r = fls_dict[r]

            for s in p_dict_r.keys():
                if s in fls_dict_r.keys():
                    X.append(fls_dict_r[s])
                    Y.append(p_dict_r[s])

    plt.plot(X, Y, marker="o", linestyle="None", alpha=0.5)

    new_x, new_y = to_average(X, Y, min_n)
    r = np.corrcoef(new_x, new_y)[0][1]

    plt.plot(new_x, new_y)

    plt.grid()

    ratio_addition = " [ratio]" if ration_flag else ""

    title = f"Final prob{ratio_addition} as a Function of FLS\nk = {k}, n = {len(X)}, r = {r}"
    plt.title(title)
    filename = f"Final prob{ratio_addition} as a Function of FLS; {k}"
    plt.savefig(filename)
    plt.clf()


def forward_success_dict(k, contin_flag=False, label_to_search=None, disappear_flag=False, n_values=None, p_threshold = 0, mlp_flag = False):
    if type(k)==list:
        fls1 =forward_success_dict(k[0], contin_flag=contin_flag, n_values=n_values, p_threshold = 0, mlp_flag = mlp_flag)
        fls2 = forward_success_dict(k[1], contin_flag=contin_flag, n_values=n_values, p_threshold=0, mlp_flag=mlp_flag)

        return minus_dicts(fls1, fls2, k[2])

    if type(k)==str:
        values_to_discreet = NUM_OF_LAYERS if n_values == None else n_values
        if k not in ["prob", "ratio prob", "increase v", "increase l"]:
            raise Exception(f"Invalid k: {k}")
        if k in ["prob", "ratio prob"]:
            ratio_flag = True if k=="ratio prob" else False
            p_dict = final_prob_dict(ratio=ratio_flag)
            return discreet_dict(p_dict, values_to_discreet)

    if k in FLS_DICT:
        return FLS_DICT[k]

    fls_dict = {}

    for relation in LABELS_DICT.keys():
        r_labels_dict = LABELS_DICT[relation]
        fls_dict[relation] = {}

        for subject in r_labels_dict.keys():
            labels = r_labels_dict[subject]
            if type(k)==int:
                labels = take_first_k(labels, k)

            true_label = labels[-1][0]

            fls = first_label_success(labels, true_label, contin_flag)
            fls_dict[relation][subject] = fls

    FLS_DICT[k] = fls_dict
    return fls_dict


def final_prob_dict(ratio=False):
    # if ration: return max/second1 else: return max

    type = "ratio" if ratio else "prob"
    if FINAL_PROBS[type]!=None:
        return FINAL_PROBS[type]

    prob_dict = {}

    for relation in prob_dict_and_labels.keys():
        r_dict = prob_dict_and_labels[relation]
        prob_dict[relation] = {}

        for subject in r_dict.keys():
            highest, second = r_dict[subject][1][0], r_dict[subject][1][1]
            p = highest / second if ratio else highest

            prob_dict[relation][subject] = p

    FINAL_PROBS[type] = prob_dict
    return prob_dict  # {relation: {subjects: prob}}


def discreet_dict(values_dict, n_values):
    if type(values_dict)==dict:
        return {key: discreet_dict(values_dict[key], n_values) for key in values_dict.keys()}
    return int(values_dict*n_values)/n_values


def fls_olu(olu_type, k, contin_flag=False, efficacy_probs=True, plot=True,
            label_to_search=None, disappear_flag=False, min_n=0, filter_dots = False, show_dots = True,
            correlation_by_average=True, n_values_to_discreet=None, return_len=False, common = False,
            min_v = 0, max_v = NUM_OF_LAYERS, spcf_key="probs plus", p_threshold=0, real_spcf = False):
    keys = ["finals", "true", "probs plus"]
    x = []
    y = {key: [] for key in keys}

    x_unusual_entropy = []
    y_unusual_entropy = {key: [] for key in keys}

    fls_dict = forward_success_dict(k, contin_flag, label_to_search, disappear_flag, n_values_to_discreet, p_threshold=p_threshold)
    if real_spcf:
        argmin_dict = get_success("argmin", "spcf", parameter=1, efficacy_prob=True, range_bound=None)
        argmin_spcf = argmin_dict[spcf_key]
    else:
        argmin_spcf = None

    olu_dict = get_success("argmax", olu_type, parameter=1, efficacy_prob=True, range_bound=argmin_spcf)

    for relation in fls_dict.keys():
        if relation in olu_dict[spcf_key]:
            fls_r_dict = fls_dict[relation]

            for subject in fls_r_dict.keys():
                if subject not in olu_dict[spcf_key][relation].keys():
                    continue

                current_v = olu_dict[spcf_key][relation][subject]
                if current_v<min_v or current_v>max_v:
                    continue

                x.append(fls_r_dict[subject])
                y[spcf_key].append(current_v)

    if not plot:
        rs = {}
        for key in keys:
            r = np.corrcoef(x, y[key])[0][1]
            new_x, new_y = to_average(x, y[key], min_n)
            if correlation_by_average:
                r = np.corrcoef(new_x, new_y)[0][1]
            if return_len:
                rs[key] = (r, len(new_x))
            else:
                rs[key] = r

        return rs

    if filter_dots:
        hist_x = {}
        for x_dot in x:
            if x_dot not in hist_x.keys():
                hist_x[x_dot] = 0
            hist_x[x_dot] += 1

        temp_x = []
        temp_y = {key:[] for key in y.keys()}
        for i in range(len(x)):
            if hist_x[x[i]] >= min_n:
                temp_x.append(x[i])
                temp_y[spcf_key].append(y[spcf_key][i])
        x = temp_x
        y = temp_y

    new_x, new_y, n = to_average(x, y[spcf_key], min_n, True, common)
    if correlation_by_average:
            r = np.corrcoef(new_x, new_y)[0][1]
    else:
            r = np.corrcoef(x, y[spcf_key])[0][1]

    plt.xlabel("FLS") if type(k) == int else plt.xlabel("final probability")
    plt.ylabel("OLU")
    if show_dots:
            plt.plot(x, y[spcf_key], "green", marker="o", linestyle="None", alpha=0.2)
            plt.plot(new_x, new_y)
            plt.plot(x_unusual_entropy, y_unusual_entropy[spcf_key], "red", marker="o", linestyle="None", alpha=0.3)
    else:
            plt.plot(new_x, new_y, marker="o", linestyle="None")
    plt.grid()
    eff_ind = "prob" if efficacy_probs else "final"

    title = f"OLU as a Function of FLS\nk: {k}, n = {n}, r = {str(r)[:4]}"
    plt.title(title)
    filename = f"OLU as a Function of FLS; {spcf_key}; {eff_ind}; {k}"
    plt.savefig(filename)
    plt.clf()


def fls_co_olu(olu_type, k, contin_flag=False, parameter=1, efficacy_probs=True, plot=True,
            label_to_search=None, disappear_flag=False, min_n=0, n_values_to_discreet=None, mlp_flag = False):
    keys = ["finals", "true", "probs plus"]
    x = None
    performance_by_fls = {key: dict() for key in keys}
    y = {key: [] for key in keys}

    fls_dict = forward_success_dict(k, contin_flag, label_to_search, disappear_flag, n_values=n_values_to_discreet, mlp_flag = mlp_flag)
    performance_dict = optimal_layer_to_update_all_dicts(olu_type, parameter, efficacy_probs, return_performance=True)

    for relation in fls_dict.keys():
        if relation in performance_dict[keys[0]]:
            fls_r_dict = fls_dict[relation]

            for subject in fls_r_dict.keys():
                if subject not in performance_dict[keys[0]][relation].keys():
                    continue

                s_fls = fls_r_dict[subject]
                for key in keys:
                    if s_fls not in performance_by_fls[key]:
                        performance_by_fls[key][s_fls] = []

                    performance_by_fls[key][s_fls].append(performance_dict[key][relation][subject])

    for key in keys:
        performance = performance_by_fls[key]
        relevant_fls = list(performance.keys())
        relevant_fls.sort()

        new_x = []

        for fls in relevant_fls:
            current_performance = performance[fls]
            if len(current_performance)<min_n:
                continue
            new_x.append(fls)
            co_olu = co_olu_one_group(current_performance)
            y[key].append(co_olu)
        if x == None:
            x = new_x

    if not plot:
        rs = {}
        for key in keys:
            r = np.corrcoef(x, y[key])[0][1]
            rs[key] = r
        return rs

    for key in keys:
        r = np.corrcoef(x, y[key])[0][1]
        plt.plot(x, y[key], marker="o", linestyle="None")

        eff_ind = "prob" if efficacy_probs else "final"

        title = f"co-OLU as a Function of FLS\nspcf: {key}; eff: {eff_ind}; k: {k}\nn = {len(x)}, r = {r}"
        plt.title(title)
        filename = f"co-OLU as a Function of FLS; {key}; {eff_ind}; {k}"
        plt.savefig(filename)
        plt.clf()


def harmonic_mean(number1, number2, factor=1):
    if 0 in [number1, number2]:
        return 0
    return (1 + factor) / ((1 / number1) + (factor / number2))


def mean(number1, number2, factor=0.0):
    return (number1*factor+number2)/(1+factor)


def success_by_layer(layer, success_type, parameter, efficacy_prob):
    if success_type not in ["spcf", "harmonic", "product", "eff"]:
        print(f"EEROR: success_type = {success_type}")
        return -1

    keys = {"finals":0, "true":2, "probs plus":4}
    success_dict = {key: {} for key in keys}
    afficacy_idx = int(efficacy_prob == True)
    layer_idx = LAYERS_RANGE.index(layer)

    for i in range(len(RESULTS_LIST)):
        res_dict = RESULTS_LIST[i]
        if res_dict==None:
            continue
        relation, subjects, _, _ = neighborhood[i]
        for key in keys.keys():
                spcf_idx = keys[key]

                if relation not in success_dict[key].keys():
                    success_dict[key][relation] = {}

                for i in range(len(subjects)):
                    subject = subjects[i]
                    efficacy = res_dict["efficacy_scores"][layer_idx][afficacy_idx][i]
                    spcf = res_dict["neighborhood_scores"][layer_idx][spcf_idx][i]

                    if type(efficacy)==list:
                        efficacy = sum(efficacy) / len(efficacy)
                    if type(spcf) == list:
                        spcf = sum(spcf) / len(spcf)
                    if key=="finals":
                        spcf=1-spcf

                    if success_type=="spcf":
                        current_success = spcf

                    elif success_type=="eff":
                        current_success = efficacy

                    else:
                        f = {"harmonic": harmonic_mean}[success_type]
                        current_success = f(spcf, efficacy, parameter)

                    success_dict[key][relation][subject] = current_success

    return success_dict


def average(l):
    return sum(l)/len(l)


def arg_min(l):
    m = (0, l[0])
    for i in range(1, len(l)):
        if l[i]<m[1]:
            m=(i, l[i])
    return m[0]


def arg_max(l):
    m = (0, l[0])
    for i in range(1, len(l)):
        if l[i]>m[1]:
            m=(i, l[i])
    return m[0]


def get_max_and_i(layer_i, success_type, parameter=1, efficacy_prob=True, min_flag=False, range_bound = None, arg_flag = False):
    # range_bound = dict{r: dict{ s: l}}
    keys = ["finals", "true", "probs plus"]

    success_dict_of_all_layer = {key: {} for key in keys}
    if arg_flag:
        f = arg_min if min_flag else arg_max
    else:
        f = min if min_flag else max

    for i in LAYERS_RANGE:
        current_success_dict = success_by_layer(i, success_type, parameter, efficacy_prob)
        for key in keys:
            for r in current_success_dict[key].keys():
                if r not in success_dict_of_all_layer[key].keys():
                    success_dict_of_all_layer[key][r] = {}
                for s in current_success_dict[key][r].keys():
                    if range_bound is not None and i>range_bound[r][s]:
                        continue
                    if s not in success_dict_of_all_layer[key][r].keys():
                        success_dict_of_all_layer[key][r][s] = []
                    success_dict_of_all_layer[key][r][s].append(current_success_dict[key][r][s])

        if i==layer_i:
            i_success_dict = current_success_dict

    max_success_dict = {key: {} for key in keys}
    for key in keys:
        for r in success_dict_of_all_layer[key].keys():
            if r not in max_success_dict[key].keys():
                max_success_dict[key][r] = {}
            for s in success_dict_of_all_layer[key][r].keys():
                max_success_dict[key][r][s] = f(success_dict_of_all_layer[key][r][s])

    return max_success_dict, i_success_dict


def minus_dicts(dict1, dict2, plus_flag=False):
    if type(dict1)==dict:
        new_dict = dict()
        for key in dict1.keys():
            if key in dict2.keys():
                new_dict[key] = minus_dicts(dict1[key], dict2[key], plus_flag)
        return new_dict
    if plus_flag:
        return dict1+dict2
    return dict1-dict2


def get_success(layer_type, success_type, parameter=1, efficacy_prob=True, range_bound = None):
    if type(layer_type)==int:
        return success_by_layer(layer_type, success_type, parameter, efficacy_prob)

    op_list = ["max", "min", "argmax", "argmin"]
    if layer_type in op_list or layer_type[-5:] == " diff":
        default_layer = 4 if layer_type in op_list else int(layer_type[:-5])
        arg_flag = layer_type[:3]=="arg"
        min_flag = layer_type[-3:]=="min"
        max_success_dict, i_success_dict = get_max_and_i(default_layer, success_type, parameter, efficacy_prob,
                                                         min_flag=min_flag, range_bound=range_bound, arg_flag=arg_flag)

        if layer_type in op_list:
            return max_success_dict
        if layer_type[-5:] == " diff":
            success_dict = minus_dicts(max_success_dict, i_success_dict)
            return success_dict

    keys = ["finals", "true", "probs plus"]

    success_dict_of_all_layer = {key: {} for key in keys}

    for i in LAYERS_RANGE:
        current_success_dict = success_by_layer(i, success_type, parameter, efficacy_prob)
        for key in keys:
          for r in current_success_dict[key].keys():
            if r not in success_dict_of_all_layer[key].keys():
                success_dict_of_all_layer[key][r] = {}
            for s in current_success_dict[key][r].keys():
                if s not in success_dict_of_all_layer[key][r].keys():
                    success_dict_of_all_layer[key][r][s]=[]
                success_dict_of_all_layer[key][r][s].append(current_success_dict[key][r][s])

    # f = {"max": max, "min": min, "mean": mean, "all": no_change, "argmax": argmax_layer}[layer_type]
    f = {"mean": mean, "all": no_change}[layer_type]

    success_dict = {key: {} for key in keys}
    for key in keys:
      for r in success_dict_of_all_layer[key].keys():
        if r not in success_dict[key].keys():
            success_dict[key][r] = {}
        for s in success_dict_of_all_layer[key][r].keys():
            success_dict[key][r][s] = f(success_dict_of_all_layer[key][r][s])

    return success_dict


def no_change(x):
    return x


def success_as_a_function_of_fls(k, layer_type, success_type, parameter=1, efficacy_prob=True, contin_flag=False, print_flag=True, dots_flag=False, min_n=0, n_values=30, mlp_flag = False, category1=-1, category2=-1, hold_on = False):
    keys = ["probs plus"]
    two_categories_flag = category1>=0 and category2>=0

    x = []
    y = {k: [] for k in keys}

    fls_dict = forward_success_dict(k, contin_flag, label_to_search=None, disappear_flag=False, n_values=n_values, mlp_flag = mlp_flag)

    success = get_success(layer_type, success_type, parameter, efficacy_prob)

    if two_categories_flag:
        dict1, dict2 = two_categories_dict(category1, category2)
        x = [[],[]]
        y = [{k: [] for k in keys}, {k: [] for k in keys}]

    if not hold_on:
        plt.figure(figsize=(10, 6))
    for rel in fls_dict.keys():
            if rel in success[keys[0]].keys():
                if two_categories_flag:
                    category_1_rel = dict1[rel]

                for s in fls_dict[rel]:
                  if s in success[keys[0]][rel].keys():

                    if two_categories_flag:
                        if s in category_1_rel:
                            x_to_add = x[0]
                            y_to_add = y[0]
                        else:
                            x_to_add = x[1]
                            y_to_add = y[1]
                    else:
                        x_to_add = x
                        y_to_add = y

                    x_to_add.append(fls_dict[rel][s])
                    for key in keys:
                        y_to_add[key].append(success[key][rel][s])

    if not print_flag:
            return x, y

    if not two_categories_flag:
            x = [x]
            y = [y]

    colours = [["blue", "green"], ["red", "black"]]
    colour_of_success_type = {"harmonic": "blue", "spcf": "orange", "eff": "green"}[success_type]
    rs = []

    key = keys[0]
    for i in range(len(x)):
                x_print, y_print = x[i], y[i]
                if dots_flag:
                    plt.plot(x_print, y_print[key], colours[0][i], marker="o", linestyle="None", alpha=0.2)
                # r = np.corrcoef(x_print, y_print[key])[0][1]
                new_x, new_y = to_average(x_print, y_print[key], min_n)

                r = np.corrcoef(new_x, new_y)[0][1]

                if hold_on:
                    success_type_name = {"harmonic": "harmonic mean", "spcf": "specificity", "eff": "efficacy"}[success_type]
                    plt.plot(new_x, new_y, colour_of_success_type, label=f"{success_type_name}; r = {str(r)[:4]}")
                else:
                    plt.plot(new_x, new_y, colours[1][i])
                plt.grid()
                rs.append(r)

    measure = "FLS" if type(k)==int else "Final Prob"
    success_name = {"harmonic": "Success", "spcf": "Specificity", "eff": "Efficacy"}[success_type]
    n = len(x[0])+len(x[1]) if len(x)>1 else len(x[0])
    title = f"{success_name} ({layer_type}) as a Function of {measure}\nk = {k}; n = {n}"
    for r in rs:
            title+=f"; r = {str(r)[:4]}"

    if hold_on:
        return n

    plt.title(title)
    plt.xlabel(measure)
    plt.ylabel("score")
    filename = f"Success in Layers as a Function of fls; {key}, {k}, {layer_type}"
    plt.savefig(filename)
    plt.clf()


def to_average(X, Y, min_n, count = False, common=False):
    unique_X = list(set(X))
    unique_X.sort()
    sums = []
    counts = []
    values = []
    for x in unique_X:
        sums.append(0)
        counts.append(0)
        values.append([])

    for i in range(len(Y)):
        y = Y[i]
        x = X[i]

        x_idx = unique_X.index(x)
        sums[x_idx]+=y
        counts[x_idx]+=1
        values[x_idx].append(y)

    for i in range(len(unique_X)-1,-1,-1):
        if counts[i]<min_n:
            del counts[i]
            del sums[i]
            del unique_X[i]
            del values[i]

    if common:
        new_y = [common_value(values[i]) for i in range(len(sums))]
    else:
        new_y = [sums[i] / counts[i] for i in range(len(sums))]

    if count:
        return unique_X, new_y, sum(counts)
    return unique_X, new_y


def to_percentage(X, Y, threshold, min_n):

    unique_X = list(set(X))
    unique_X.sort()
    counts_relevant = []
    counts = []
    for x in unique_X:
        counts_relevant.append(0)
        counts.append(0)

    for i in range(len(Y)):
        y = Y[i]
        x = X[i]
        x_idx = unique_X.index(x)
        counts_relevant[x_idx]+=int(y<threshold)
        counts[x_idx]+=1

    for i in range(len(unique_X)-1,-1,-1):
        if counts[i]<min_n:
            del counts[i]
            del counts_relevant[i]
            del unique_X[i]

    return unique_X, [counts_relevant[i]*100 / counts[i] for i in range(len(counts_relevant))]


def percentage_of_low_success(k, layer_type, success_type, parameter=1, threshold_range = [i/10 for i in range(1,10)], efficacy_prob=True, contin_flag=False, spcf_types=[], min_n=0, n_values=None, mlp_flag=False):
    x, y = success_as_a_function_of_fls(k, layer_type, success_type, parameter, efficacy_prob, contin_flag,
                                        print_flag=False, n_values=n_values, mlp_flag=mlp_flag)
    general_title = f"percentage of proximity to max" if type(layer_type)==str and layer_type[-4:]=="diff" else f"percentage of low success ({success_type})"
    layer_str = layer_type[:-5] if type(layer_type)==str and layer_type[-4:]=="diff" else layer_type
    title = general_title + f"\nk={k}, layer ={layer_str}"
    general_filename = f"{general_title}. {k}. {layer_type}. min n = {min_n}"
    for spcf_type in spcf_types:
        fig, ax = plt.subplots()
        y_key = y[spcf_type]
        for threshold in threshold_range:
            temp_x, temp_y = to_percentage(x, y_key, threshold, min_n=min_n)
            ax.plot(temp_x, temp_y, label=str(threshold))

        ax.legend()
        ax.grid()
        plt.xlabel("FLS") if type(k) == int else plt.xlabel("final probability")
        plt.ylabel("%")

        current_title = title
        if len(threshold_range)==1:
            r = np.corrcoef(temp_x, temp_y)[0][1]
            current_title += f", r = {r}"
        plt.title(current_title)
        filename = general_filename+f", {spcf_type}.png"
        plt.savefig(filename)
        plt.clf()


def success_diff_dict(layer1, layer2, success_type, efficacy_prob=True):
    success1 = get_success(layer1, success_type, 1, efficacy_prob)
    success2 = get_success(layer2, success_type, 1, efficacy_prob)
    return minus_dicts(success1, success2)


def percentage_of_OLU_lower_than(k, success_type, threshold_layer = 15, efficacy_prob=True, spcf_types=["probs plus"], min_n=0, n_values=None, p_threshold=0, mlp_flag = False):
    # percentage for OLU < threshold_layer (if threshold_layer is int) or score[threshold_layer[0]] < score[threshold_layer[1]] (if it is a couple).
    x = []
    y = {k: [] for k in spcf_types}

    fls_dict = forward_success_dict(k, contin_flag=False, label_to_search=None, disappear_flag=False, n_values=n_values, p_threshold=p_threshold, mlp_flag = mlp_flag)
    if type(threshold_layer)==list:
        olu_dict = success_diff_dict(threshold_layer[0], threshold_layer[1], success_type, efficacy_prob=True)
    else:
        olu_dict = optimal_layer_to_update_all_dicts(success_type, efficacy_threshold=1, efficacy_probs=efficacy_prob)

    for rel in fls_dict.keys():
        if rel in olu_dict[spcf_types[0]].keys():
            for s in fls_dict[rel]:
              if s in olu_dict[spcf_types[0]][rel].keys():
                x.append(fls_dict[rel][s])
                for key in spcf_types:
                    y[key].append(olu_dict[key][rel][s])

    for key in spcf_types:
        y_key = y[key]
        if type(threshold_layer) == list:
            temp_x, temp_y = to_percentage(x, y_key, 0, min_n=min_n)
            r = np.corrcoef(temp_x, temp_y)[0][1]
            filename = f"percentage of score[{threshold_layer[1]}] greater than score[{threshold_layer[0]}]; {success_type}; {key}"
            title = f"Percentage of score[{threshold_layer[1]}] greater than score[{threshold_layer[0]}]\nk = {k}, r = {str(r)[:4]}"
        else:
            temp_x, temp_y = to_percentage(x, y_key, threshold_layer, min_n=min_n)
            r = np.corrcoef(temp_x, temp_y)[0][1]
            filename = f"percentage of OLU lower than {threshold_layer}; {success_type}; {key}"
            title = f"Percentage of OLU lower than {threshold_layer}\nk = {k}, r = {str(r)[:4]}"
        plt.plot(temp_x, temp_y, marker="o", linestyle="None")
        plt.xlabel("FLS") if type(k) == int else plt.xlabel("final probability")
        plt.ylabel("%")
        plt.grid()

        plt.title(title)
        plt.savefig(filename)
        plt.clf()


def is_better(soft_v, rand_v, p_to_improve, old_val):
    if rand_v == 0: return False
    if p_to_improve=="prod":
        new_prod = soft_v**2 / rand_v
        return new_prod>old_val
    return soft_v>p_to_improve and soft_v / rand_v > old_val


def co_olu_one_group(lists):
    mean_list = [sum([l[i] for l in lists])/len(lists) for i in range(len(lists[0]))]
    olu_idx = np.argmax(mean_list)
    olu = LAYERS_RANGE[olu_idx]
    return olu


def harmonic_mean_list(list1, list2):
    new_list = []
    for i in range(len(list1)):
        number1, number2 = list1[i], list2[i]
        if 0 in [number1, number2]:
            new_number = 0
        else:
            new_number = 2 / ((1 / number1) + (1 / number2))
        new_list.append(new_number)
    return new_list


def average_lists(lists, final_avrg = False, by_rows = False, dont_change = False):
    if final_avrg:
        return average([average(l) for l in lists])
    elif by_rows:
        return [sum(l) / len(l) for l in lists]
    elif dont_change:
        return lists
    else:
        return [sum([l[i] for l in lists])/len(lists) for i in range(len(lists[0]))]


def change_list_by_factors(values_list, factors = [], ops = [] ):
    new_list = []
    for v in values_list:

        ops_dict = {"plus": lambda x, y: x+y,
                "minus": lambda x, y: x-y,
                "minus reverse": lambda x, y: y - x,
                "div": lambda x, y: x/y,
                "mult": lambda x, y: x*y}
        result = v
        for i in range(len(factors)):
            result = ops_dict[ops[i]](result, factors[i])
        new_list.append(result)
    return new_list


def percentile(values_list, per=0.25):
    values_list.sort()
    index = int(per * len(values_list))
    return values_list[index]


def calc_inter_fraction(lists, per=0.25):
    per_up = 1-per

    lower_list = []
    upper_list = []

    print(len(lists))

    for i in range(len(lists[0])):
        i_values = [l[i] for l in lists]
        lower_value = percentile(i_values, per)
        upper_value = percentile(i_values, per_up)
        lower_list.append(lower_value)
        upper_list.append(upper_value)
    return lower_list, upper_list


def std(l):
    avrg = average(l)
    var = sum([(x-avrg)**2 for x in l])/len(l)
    return math.sqrt(var)


def fls_extremum_spcf(k, min_n=0, show_dots = True,
            correlation_by_average=True, n_values_to_discreet=None, min_v = 0, max_v = NUM_OF_LAYERS,
            spcf_key="probs plus", p_threshold=0, succes_instead_of_spcf=False):

    argmin_spcf_g = {"x": [], "y": [], "name": "Argmin Specificity"}
    argmax_name = "Argmax Real Success" if succes_instead_of_spcf else "Argmax Real Specificity"
    argmax_real_spcf_g = {"x": [], "y": [], "name": argmax_name}
    min_spcf_g = {"x": [], "y": [], "name": "Min Specificity"}

    fls_dict = forward_success_dict(k, n_values=n_values_to_discreet, p_threshold=p_threshold)

    argmin_dict = get_success("argmin", "spcf", parameter=1, efficacy_prob=True, range_bound=None)
    argmin_spcf = argmin_dict[spcf_key]
    if succes_instead_of_spcf:
        argmax_real_dict = get_success("argmax", "harmonic", parameter=1, efficacy_prob=True, range_bound=argmin_spcf)
    else:
        argmax_real_dict = get_success("argmax", "spcf", parameter=1, efficacy_prob=True, range_bound=argmin_spcf)
    argmax_real_spcf = argmax_real_dict[spcf_key]
    min_dict = get_success("min", "spcf", parameter=1, efficacy_prob=True, range_bound=None)
    min_spcf = min_dict[spcf_key]


    for relation in argmin_spcf.keys():
        if relation in fls_dict.keys():
            fls_r_dict = fls_dict[relation]

            for subject in fls_r_dict.keys():
                if subject not in argmin_spcf[relation].keys():
                    continue

                current_argmin = argmin_spcf[relation][subject]
                current_argmax_real = argmax_real_spcf[relation][subject]
                current_min = min_spcf[relation][subject]

                if current_argmin>=min_v and current_argmin<=max_v:
                    argmin_spcf_g["x"].append(fls_r_dict[subject])
                    argmin_spcf_g["y"].append(current_argmin)

                if current_argmin>=min_v and current_argmin<=max_v:
                    argmax_real_spcf_g["x"].append(fls_r_dict[subject])
                    argmax_real_spcf_g["y"].append(current_argmax_real)

                if current_argmin>=min_v and current_argmin<=max_v:
                    min_spcf_g["x"].append(fls_r_dict[subject])
                    min_spcf_g["y"].append(current_min)

    for graph in [argmin_spcf_g, argmax_real_spcf_g, min_spcf_g]:

         x, y = graph["x"], graph["y"]

         new_x, new_y, n = to_average(x, y, min_n, True)
         if correlation_by_average:
               r = np.corrcoef(new_x, new_y)[0][1]
         else:
               r = np.corrcoef(x, y)[0][1]

         if show_dots:
                 plt.plot(x, y, "green", marker="o", linestyle="None", alpha=0.2)
                 plt.xlabel("FLS") if type(k) == int else plt.xlabel("final probability")
                 plt.ylabel("OLU")
                 plt.plot(new_x, new_y)
         else:
                 plt.plot(new_x, new_y, marker="o", linestyle="None")
         plt.grid()

         title = f"{graph['name']} as a Function of FLS\nk: {k}, n = {n}, r = {str(r)[:4]}"
         plt.title(title)
         filename = f"{graph['name']} as a Function of FLS; {spcf_key}; {k}"
         plt.savefig(filename)
         plt.clf()


def scores_as_a_function_of_fls(k, layer_type, min_n=0, n_values=30):
    for success_type in ["harmonic", "spcf", "eff"]:
        n = success_as_a_function_of_fls(k, layer_type, success_type, parameter=1, efficacy_prob=True,
                                         contin_flag=False, print_flag=True, dots_flag=False, min_n=min_n,
                                         n_values=n_values, mlp_flag=False, category1=-1, category2=-1, hold_on=True)

    measure = "FLS" if type(k) == int else "Final Prob"
    title = f"Score ({layer_type}) as a Function of {measure}\nn = {n}"
    if measure=="FLS":
        title+= f"; k = {k}"
    plt.title(title)
    plt.legend()
    plt.xlabel(measure)
    plt.ylabel("score")
    filename = f"Score in Layers as a Function of fls; {k}, {layer_type}"
    plt.savefig(filename)
    plt.clf()


########
# RUNS #
########

## Percentage of low success (score < threshold) as a function of FLS or final probability (depending on k)
percentage_of_low_success(k, "1 diff", "harmonic", threshold_range=[0.05, 0.1], spcf_types=["probs plus"], min_n=min_n, n_values=n_values)

## percentage of:
## if threshold_layer is int: OLU < threshold_layer
## if threshold_layer is [int1, int2]: sore[int1] < score[int2]
percentage_of_OLU_lower_than(1, "harmonic", threshold_layer=[0, 20], spcf_types=["probs plus"], min_n=min_n, n_values=n_values)

## score (harmonic mean, efficacy, specificity) as a function of FLS or final probability (depending on k)
scores_as_a_function_of_fls("prob", 1, min_n, n_values)

## OLU as a function of FLS or final probability (depending on k)
fls_olu(olu_type="harmonic", k=k, show_dots=False, n_values_to_discreet=n_values, common=False, real_spcf=False)

## co-OLU (the layer where the average score of all prompts in the set) as a function of FLS or final probability (depending on k)
fls_co_olu("harmonic", k, min_n=min_n, n_values_to_discreet=n_values)

## Extremum of spcf as a function of FLS
fls_extremum_spcf("prob", min_n=min_n, show_dots=False, correlation_by_average=True, n_values_to_discreet=n_values, spcf_key="probs plus", succes_instead_of_spcf=False)
