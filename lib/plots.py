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
    print(activities_ordered)
    plt.bar(indices_ordered[min_index:], activity_acc_sorted[min_index:])
    #plt.plot([i for i in range(len(accuracies_per_activity))], accuracies_per_activity, '-o')
    plt.savefig("figures/acc_per_activity")
