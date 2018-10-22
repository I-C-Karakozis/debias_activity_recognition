import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def plot_accuracy_per_activity(accuracies_per_activity, encoder):
    indices_ordered = numpy.argsort(accuracies_per_activity)
    acitivities_ordered = [encoder.decode_noun(index) for index in indices_ordered]
    activity_acc_sorted = numpy.sort(accuracies_per_activity)
    plt.bar(indices_ordered, accuracies_per_activity)
    #plt.plot([i for i in range(len(accuracies_per_activity))], accuracies_per_activity, '-o')
    plt.savefig("figures/acc_per_activity")
