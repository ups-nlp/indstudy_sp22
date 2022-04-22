import matplotlib.pyplot as plt

# constructs a plot with the given data
def construct_plot(x_arr, y_arr, x_label, y_label, title):
    plt.figure()
    plt.plot(x_arr,y_arr)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
