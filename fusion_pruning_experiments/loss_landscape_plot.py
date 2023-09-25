import json 
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, interp2d
from scipy.ndimage import gaussian_filter

if __name__ == '__main__':

    results_file = "./fusion_pruning_experiments/results_of_loss_landscape_Cifar10_vgg11_grid60_sp40_l1_margin50.json"
    with open(results_file, "r") as file:
        results_dict = json.load(file)

    list_of_2D_coords = []
    list_of_perf = []
    for point in results_dict["grid"]:
        this_2D = point["model_2D_vec"]
        this_perf = point["model_perf"]
        list_of_2D_coords.append(this_2D)
        list_of_perf.append(this_perf)

    orig_model_2D = results_dict["original"]["model_2D_vec"]
    orig_perf = results_dict["original"]["model_perf"]
    print("Original Model:")
    print(orig_perf)

    pr_model_2D = results_dict["pruned"]["model_2D_vec"]
    pr_perf = results_dict["pruned"]["model_perf"]
    print("Pruned Model:")
    print(pr_perf)

    if_model_2D = results_dict["intra_fusion"]["model_2D_vec"]
    if_perf = results_dict["intra_fusion"]["model_perf"]
    print("Intra-Fusion Model:")
    print(if_perf)

    x_coordinates, y_coordinates = zip(*list_of_2D_coords)

    # Define the range and density of points for the grid
    x_min, x_max, y_min, y_max = min(x_coordinates), max(x_coordinates), min(y_coordinates), max(y_coordinates)
    num_points_x, num_points_y = 600, 600  # Adjust density as needed

    # Create a grid of points
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_points_x), np.linspace(y_min, y_max, num_points_y))

    # Spline interpolation
    interp_func = interp2d(x_coordinates, y_coordinates, list_of_perf, kind='cubic')
    performance_grid_spline = interp_func(x_grid[0], y_grid[:, 0])

    # Apply Gaussian smoothing
    sigma = 3 # Adjust the standard deviation as needed
    performance_grid_bilinear_smoothed = gaussian_filter(performance_grid_spline, sigma)

    # Visualization (optional)
    sc = plt.imshow(performance_grid_bilinear_smoothed, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='RdYlBu_r')
    
    #plt.colorbar()
    #plt.savefig("loss_landscape_plot.png")
    #plt.show()

    #sc = plt.scatter(*zip(*list_of_2D_coords), c=list_of_perf, cmap='RdYlBu_r', marker='s', s=15)
    cbar = plt.colorbar(sc, label='Accuracy')
    
    #list_of_2D_coords.extend([orig_model_2D, pr_model_2D, if_model_2D])
    #list_of_perf.extend([orig_perf, pr_perf, if_perf])
    #given_models_2D = [orig_model_2D, pr_model_2D, if_model_2D]
    #given_models_perf = [orig_perf, pr_perf, if_perf]
    #plt.scatter(*zip(*given_models_2D), marker='o', c="black", s=60)

    # Create a scatter plot with the points and annotations
    points = [(orig_model_2D[0], orig_model_2D[1], f"Original Model: {round(orig_perf*100)}%"),
              (pr_model_2D[0], pr_model_2D[1], f"Pruned Model: {round(pr_perf*100)}%"),
              (if_model_2D[0], if_model_2D[1], f"Intra-Fusion Model: {round(if_perf*100)}%")]
    cnt = 0
    for x, y, label in points:
        plt.scatter(x, y, marker='o', s=50, label=label, c="black")
        if cnt != 2:
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(-5, 10), ha='center', weight='bold')
        else:
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(22, -18), ha='center', weight='bold')
        cnt += 1


    plt.savefig("./loss_landscape_plot.png")
    plt.show()
    
