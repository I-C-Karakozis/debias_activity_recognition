import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def plot_accuracy_per_activity(accuracies_per_activity, encoder):
    indices_ordered = numpy.argsort(accuracies_per_activity)
    activities_ordered = [encoder.decode_verb(index) for index in indices_ordered]
    activity_acc_sorted = numpy.sort(accuracies_per_activity).tolist()
    min_index = len(activity_acc_sorted) - activity_acc_sorted[::-1].index(-1)
    shifted_indices = [i for i in range(0, len(activity_acc_sorted)-min_index)]

    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(shifted_indices, activity_acc_sorted[min_index:])
    ax.set_yticks(shifted_indices)
    ax.set_yticklabels(activities_ordered[min_index:])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Activity Cls Accuracy')
    #plt.bar(shifted_indices, activity_acc_sorted[min_index:])
    #plt.xticks(shifted_indices, activities_ordered)
    plt.savefig("figures/acc_per_activity")
