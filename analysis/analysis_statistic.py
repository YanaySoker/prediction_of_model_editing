# from analysis_new import *
# from analysis_ll import *
from neighborhood import d as neighborhood
from scores_0_6__7_24__26_51__52_80__82_86__88_102 import d as results_list
import copy
from analysis_config import *
import numpy as np
import matplotlib.pyplot as plt

RESULTS_LIST = results_list
LAYERS_RANGE = HIGH_RES_LAYERS_RANGE

PRINTED = [False]

def harmonic_mean(number1, number2):
    if 0 in [number1, number2]:
        return 0
    return 2 / ((1 / number1) + (1 / number2))


def average(single_list):
    return sum(single_list) / len(single_list)


def average_lists(lists, by="columns"):
    if by=="columns":
        return [average([lists[j][i] for j in range(len(lists))]) for i in range(len(lists[0]))]
    if by=="rows":
        return [average(single_list) for single_list in lists]
    if by=="all":
        return average([average(single_list) for single_list in lists])


def flip_all(lists):
    if type(lists)==list:
        new_lists = copy.deepcopy(lists)
        for i in range(len(lists)):
            new_lists[i]=flip_all(lists[i])
        return new_lists
    elif type(lists)==dict:
        new_lists = copy.deepcopy(lists)
        for i in lists.keys():
            new_lists[i]=flip_all(lists[i])
        return new_lists
    else:
        return 1-lists


def olu_lists(r_list, olu_type, spcf_type, efficacy_type, min_size=0, return_all_scores = False, cross_editing=False):
    # return 2 lists: relation olus; subject olus

    spcf_idx = {"final":0, "true": 2, "plus": 4}[spcf_type]
    eff_idx = {"final":0, "prob": 1}[efficacy_type]

    all_scores = {}
    f = {"harmonic": harmonic_mean, "product": lambda x,y: x*y}[olu_type]

    local_results_list = RESULTS_LIST

    for res_idx in range(len(local_results_list)):
        r_name = neighborhood[res_idx][0]
        if r_name not in r_list:
            continue
        results_dict = local_results_list[res_idx]
        if results_dict==None:
            continue

        if r_name not in all_scores.keys():
            all_scores[r_name] = []

        large_spcf_scores = results_dict["neighborhood_scores"]    # n_layers x types x n_sbjects {x n_objects}
        n_layers = len(large_spcf_scores)
        n_subjects = len(large_spcf_scores[0][0])

        spcf_scores = [[large_spcf_scores[l][spcf_idx][s] for s in range(n_subjects)] for l in range(n_layers)]

        large_eff_scores = results_dict["efficacy_scores"]
        eff_scores = [[large_eff_scores[l][eff_idx][s] for s in range(n_subjects)] for l in range(n_layers)]

        for sub_idx in range(n_subjects):
            current_spcf = [spcf_scores[l][sub_idx] for l in range(n_layers)]
            current_eff = [eff_scores[l][sub_idx] for l in range(n_layers)]

            if spcf_type=="final":
                current_spcf = flip_all(current_spcf)

            scores = [f(current_spcf[i], current_eff[i]) for i in range(n_layers)]
            all_scores[r_name].append(scores)

    olus_relations_list = []
    olus_subjects_list = []

    if return_all_scores:
        return all_scores   # {rel_name: [[], [], ..., []]}

    count_valid = 0

    for r_name in all_scores.keys():
        scores = all_scores[r_name]

        avrg_scores = average_lists(scores, "columns")
        max_idx = np.argmax(avrg_scores)

        if len(scores)>=min_size:
            olus_relations_list.append(LAYERS_RANGE[max_idx])
            count_valid+=1

        for s_idx in range(len(scores)):
                max_idx = np.argmax(scores[s_idx])
                olus_subjects_list.append(LAYERS_RANGE[max_idx])

    if not PRINTED[0]:
        print("Number of relations =", len(all_scores))
        print("Number of large relations =", count_valid)
        PRINTED[0]=True

    return olus_relations_list, olus_subjects_list


def print_statistics(r_list, min_size=0):
    # "product"
    for olu_type in ["harmonic"]:
        for spcf_type in ["plus"]:
            for efficacy_type in ["prob"]:
                if [spcf_type, efficacy_type].count("final")==1:
                    continue
                olus_relations_list, olus_subjects_list = olu_lists(r_list, olu_type, spcf_type, efficacy_type, min_size)
                print(f"\nOLU type = {olu_type}\nEfficacy type = {efficacy_type}\nSpecificity type = {spcf_type}")

                count_by_relations = []
                count_by_subjects = []

                for layer in LAYERS_RANGE:
                    count_by_relations.append(olus_relations_list.count(layer))
                    count_by_subjects.append(olus_subjects_list.count(layer))

                spcf_name = "prob" if spcf_type=="true" else spcf_type
                file_name_prompt = f"OLU histogram by prompts, {olu_type}, {efficacy_type}, {spcf_name}.png"
                title_prompt = f"OLU histogram of prompts\nn = {sum(count_by_subjects)}"

                common_ixd = np.argmax(count_by_subjects)
                common_value = LAYERS_RANGE[common_ixd]

                frac = count_by_subjects[common_ixd] / sum(count_by_subjects)
                print(f"common OLU by prompts = {common_value}\t({frac*100}%)")

                plt.plot(LAYERS_RANGE, count_by_subjects)
                plt.title(title_prompt)
                plt.savefig(file_name_prompt)
                plt.clf()

                file_name_relation = f"OLU histogram by relations, {olu_type}, {efficacy_type}, {spcf_name}.png"
                title_relation = f"OLU histogram by relations\nn = {len(count_by_relations)}"

                common_ixd = np.argmax(count_by_relations)
                common_value = LAYERS_RANGE[common_ixd]

                frac = count_by_subjects[common_ixd] / sum(count_by_subjects)
                print(f"common OLU by relations = {common_value} \t({round(frac * 100)}%)")

                plt.plot(LAYERS_RANGE, count_by_relations)
                plt.title(title_relation)
                plt.savefig(file_name_relation)
                plt.clf()


def histogram_of_max_lists(lists, norm=True):
    hist = [0] * len(LAYERS_RANGE)
    for list in lists:
        max_idx = np.argmax(list)
        hist[LAYERS_RANGE[max_idx]]+=1
    if norm:
        s = sum(hist)
        hist = [v/s for v in hist]
    return hist


def relations_histograms(r_list, norm=True, cross_editing = False):
    for olu_type in ["harmonic"]:
        for spcf_type in ["final", "true", "plus"]:
            for efficacy_type in ["final", "prob"]:
                if [spcf_type, efficacy_type].count("final")==1:
                    continue
                all_scores = olu_lists(r_list, olu_type, spcf_type, efficacy_type, min_size=0, return_all_scores=True, cross_editing=cross_editing)
                print(f"\nOLU type = {olu_type}\nEfficacy type = {efficacy_type}\nSpecificity type = {spcf_type}")

                max_len = max([len(v) for v in all_scores.values()])
                all_hist = []
                all_scores_subjects = []
                for rel in all_scores.keys():
                    rel_hist = histogram_of_max_lists(all_scores[rel], norm)
                    alpha = len(all_scores[rel])/max_len
                    plt.plot(LAYERS_RANGE, rel_hist, "blue", alpha=alpha)
                    all_hist.append(rel_hist)
                    all_scores_subjects+=all_scores[rel]
                mean_hist_rel = average_lists(all_hist)
                # mean_scores = average_lists(all_scores_subjects)

                plt.plot(LAYERS_RANGE, mean_hist_rel, "black", label="mean histogram")
                plt.legend()
                plt.grid()

                spcf_name = "prob" if spcf_type == "true" else spcf_type
                title = f"OLU histogram of prompts by relations, {efficacy_type}, {spcf_name}"

                plt.title(title)
                if cross_editing:
                    title+=" cross editing"
                plt.savefig(title+".png")
                plt.clf()


def average(l):
    try:
        return sum(l)/len(l)
    except:
        return "ERROR"


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


def print_success_by_relations(relation_set=[], success_type="harmonic", specifisity_name="prob", by_probs=True, name="", size_limit=None, transparency=True):
    # specifisity types:
    # 0: "finals"
    # 2: "prob"
    # 4: "plus"

    specifisity_type = {"finals":0, "prob":2, "plus":4}[specifisity_name]

    to_flip = int(specifisity_type in [0, 3])

    to_print = [i for i in range(len(RESULTS_LIST)) if RESULTS_LIST[i] is not None]
    if len(relation_set) > 0:
        to_print = [i for i in to_print if i in relation_set]
    if size_limit:
        to_print = [i for i in to_print if len(neighborhood[i][1]) <= size_limit]

    relations_list = [nghbr[0] for nghbr in neighborhood]
    if [] in relations_list:
        relations_list.remove([])
    relations_list = set(relations_list)
    relations_list = list(relations_list)

    results_by_relation = [[] for _ in range(len(relations_list))]
    results_all = []

    for i in to_print:
        relation_idx = relations_list.index(neighborhood[i][0])

        results_dict = RESULTS_LIST[i]
        spcf_scores = results_dict["neighborhood_scores"]
        eff_scores = results_dict["efficacy_scores"]
        eff_type = int(by_probs)

        count = 0
        for subj_idx in range(len(spcf_scores[0][0])):
            count += 1
            specifisity = [(spcf_scores[L][specifisity_type][subj_idx] - to_flip) * (-1) ** to_flip for L in
                           range(len(LAYERS_RANGE))]
            efficasy = [eff_scores[L][eff_type][subj_idx] for L in range(len(LAYERS_RANGE))]

            new_result_list = {"harmonic": harmonic_mean_list(specifisity, efficasy), "spcf": specifisity, "eff": efficasy}[success_type]
            results_by_relation[relation_idx].append(new_result_list)
            results_all.append(new_result_list)

    mean_results_by_relations = []
    for i in range(len(results_by_relation)):
        rel_results = results_by_relation[i]
        if len(rel_results)==0:
            continue

        average_results = average_lists(rel_results)
        mean_results_by_relations.append(average_results)

    mean_over_prompts = average_lists(results_all)

    plt.plot(LAYERS_RANGE, mean_over_prompts, "black", label="average over prompts")
    plt.legend()

    max_idx_p = np.argmax(mean_over_prompts)
    max_layer_p = LAYERS_RANGE[max_idx_p]


    print(f"mean OLU: by prompts = {max_layer_p}")

    t = 'probs' if by_probs else 'finals'
    title = {"harmonic": "Harmonic Mean", "spcf": "Specificity", "eff": "Efficacy"}[success_type]
    plt.title(title)
    plt.grid()

    filename = "success_by_layers_"

    filename += f"{t}_{specifisity_name}.png"
    plt.savefig(filename)
    plt.clf()


def average_lists(lists, final_avrg = False, by_rows = False, dont_change = False):
    if final_avrg:
        return average([average(l) for l in lists])
    elif by_rows:
        return [sum(l) / len(l) for l in lists]
    elif dont_change:
        return lists
    else:
        return [sum([l[i] for l in lists])/len(lists) for i in range(len(lists[0]))]


def avrg_olu_exp_dict(olu_exp_dict, key, efficacy_probs, efficacy_flag):
    all_scores = olu_exp_dict["efficacy_scores"] if efficacy_flag else olu_exp_dict["neighborhood_scores"]
    print("all_scores len:", len(all_scores))
    n_subjects = len(all_scores[0])
    if efficacy_flag:
        key_idx = int(efficacy_probs)
    else:
        key_idx = {"final": 0, "new": 1, "true": 2, "plus": 4}[key]
    scores = all_scores[key_idx]  # size: n_subjects
    olu_sum = sum(scores)
    return olu_sum, n_subjects


# functions:
relations_histograms(train_set, cross_editing=False)
print_statistics(train_set+test_set, min_size=0)
print_success_by_relations(relation_set=range(125), success_type = "harmonic", specifisity_name="plus", by_probs=True, name="", size_limit=None, transparency=True)
# specifisity types: "finals"; "prob"; "plus"
# success_type: "harmonic"; "spcf"; "eff"
