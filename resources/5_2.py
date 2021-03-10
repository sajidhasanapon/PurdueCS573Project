import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt


def train(train_set):
    """
    train_set is a dataframe
    """
    yes_set =   train_set[train_set["decision"]==1]
    no_set =    train_set[train_set["decision"]==0]

    yes_prob =  len(yes_set) / (len(train_set) * 1.0)
    no_prob =   len(no_set) / (len(train_set) * 1.0)

    yes_dict = {}
    no_dict = {}

    L = [x for x in train_set.columns]
    L.remove("decision")
    for attr_name in L:
        yes_dict[attr_name] = {}
        no_dict[attr_name] = {}

        vals = train_set[attr_name].unique()
        for val in vals:
            yes_dict[attr_name][val] = (len(yes_set[yes_set[attr_name] == val]) + 1.0) / (len(yes_set) + len(vals))
            no_dict[attr_name][val] = (len(no_set[no_set[attr_name] == val]) + 1.0) / (len(no_set) + len(vals))

    return yes_prob, no_prob, yes_dict, no_dict



def fit(data, yes_prob, no_prob, yes_dict, no_dict):
    """
    data is a dataframe
    """

    cnt_correct = 0
    L = [x for x in data.columns]
    L.remove("decision")

    for tuple in data.itertuples():
        logL_yes = 0.0
        logL_no = 0.0

        for attr_name in L:
            val = getattr(tuple, attr_name)
            logL_yes += math.log(yes_dict[attr_name][val]) if val in yes_dict[attr_name].keys() else 0.0 # log(1) = 0
            logL_no += math.log(no_dict[attr_name][val]) if val in no_dict[attr_name].keys() else 0 #log(1) = 0
        
        logL_yes += yes_prob
        logL_no  += no_prob

        decision = 1 if logL_yes >= logL_no else 0
        if decision == getattr(tuple, "decision"):
            cnt_correct += 1

    return (cnt_correct) / len(data)

computed_before = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests", "attractive_important", "sincere_important", "intelligence_important", "funny_important", "ambition_important", "shared_interests_important"]
L = ["age", "age_o", "importance_same_race", "importance_same_religion", "pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests", "attractive_important", "sincere_important", "intelligence_important", "funny_important", "ambition_important", "shared_interests_important", "attractive", "sincere", "intelligence", "funny", "ambition", "attractive_partner", "sincere_partner", "intelligence_partner", "funny_partner", "ambition_partner", "shared_interests_partner", "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga", "interests_correlate", "expected_happy_with_sd_people", "like"]


B = [2, 5, 10, 50, 100, 200]
list_train_acc = []
list_test_acc = []

for num_bins in B:
    df = pd.read_csv("dating.csv")
    for attr_name in L:
        minimum = 0.0
        maximum = 10.0
        if attr_name in ["age", "age_o"]:
            minimum = 18.0
            maximum = 58.0
        elif attr_name in computed_before:
            maximum = 1.0
        elif attr_name == "interests_correlate":
            minimum = -1.0
            maximum = 1.0
        
        df[attr_name] = df[attr_name].map(lambda prev_val: int(max(0, min(num_bins-1.0, math.ceil(num_bins*(prev_val - minimum) / (maximum - minimum))-1))))



    test_set = df.sample(frac=0.2,random_state=47)
    train_set = df.drop(test_set.index)

    yes_prob, no_prob, yes_dict, no_dict = train(train_set)
    
    train_acc = fit(train_set, yes_prob, no_prob, yes_dict, no_dict)
    test_acc = fit(test_set, yes_prob, no_prob, yes_dict, no_dict)

    print("Bin Size:\t%d" %(num_bins))
    print("Training Accuracy:\t%0.2f" %(train_acc))
    print("Testing Accuracy:\t%0.2f" %(test_acc))
    list_train_acc.append(train_acc)
    list_test_acc.append(test_acc)


plot_df = pd.DataFrame({"Train": list_train_acc, "Test": list_test_acc}, index=B)
ax = plot_df.plot.line()
ax.set_xlabel("b")
ax.set_ylabel("Accuracy")
ax.set_xticks(B)
plt.savefig("5_2.pdf", bbox_inches="tight")
