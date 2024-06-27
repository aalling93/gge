import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, List
import matplotlib.colors as mcolors


class SOM:
    r"""
    Self-Organizing Maps (SOMs), also known as Kohonen maps, are a type of unsupervised learning algorithm that
    uses neural networks for data visualization by reducing dimensions, typically transforming complex,
    multidimensional data into simpler, lower-dimensional representations. These representations often retain
    crucial topological and metric relationships of the original data spaces, which is advantageous for
    visualizing high-dimensional data sets.

    SOMs consist of components called nodes or neurons, and they are positioned on a regular, usually
    two-dimensional grid. Each neuron in the grid is fully connected to the input layer. The training involves
    adjusting these connections (weights) based on input data. The goal of the training is to spatially organize
    the input data onto the grid such that similar data items are mapped to adjacent neurons, preserving the
    topology of the data set.

    The training process uses competitive learning. When an input vector is presented, the neuron with the
    weight vector most similar to the input vector (the Best Matching Unit, BMU) is determined. The weights of
    the BMU and neurons within its neighborhood are adjusted towards the input vector. The size of the
    neighborhood and the learning rate decrease over time.


    Change/Anomaly Detection, e.g., for glachiers or other environmental data:
    =========================
    In anomaly detection, SOMs identify outliers by analyzing the quantization errors and the density of hits
    per neuron. Neurons representing few or no data points, or those far from their neighbors, may indicate
    anomalies. For change detection, SOMs trained on data from different time periods can highlight shifts
    in data characteristics, with significant changes in neuron weights suggesting underlying changes in the
    dataset.


    The key to effective anomaly and change detection with SOMs lies in their ability to reduce data dimensionality
    while maintaining the intrinsic structure, making it easier to spot outliers or changes visually and
    algorithmically.



    Mathematical Explanation:
    =========================
    Each neuron \(i\) in the map is associated with a weight vector \( w_i \in R^n \), where \( n \) is the
    dimensionality of the input data. The SOM learning involves iterating through the training data and
    updating the weights according to the following rules:

    1. **Finding the BMU**:
       The BMU for an input vector \( x \) is the neuron whose weight vector is closest to \( x \), typically
       using Euclidean distance:
       \[
       BMU(x) = \arg\min_i || x - w_i ||
       \]

    2. **Updating Weights**:
       The weights of the BMU and its neighbors are adjusted towards \( x \):
       \[
       w_i(t+1) = w_i(t) + \theta(i, BMU, t) \cdot \alpha(t) \cdot (x - w_i(t))
       \]
       where \( \theta(i, BMU, t) \) is a neighborhood function centered around the BMU, typically Gaussian,
       and \( \alpha(t) \) is the learning rate, which decreases over time.

    3. **Continuation**:
       These steps are repeated for each input vector for a number of epochs, gradually organizing the map
       and reducing the learning rate and neighborhood size.

    Analysis:
    =========
    After training, the map can be used to visualize high-dimensional data in two dimensions. Each neuron
    represents a cluster of similar data items, and the physical location of a neuron in the map corresponds
    to intrinsic properties of its associated data items. Analyzing the trained map can reveal relationships
    and patterns in the data, such as clustering and outliers.

    Extremes:
    =========
    In extreme cases, such as when the learning rate is too high or too low, the map may not organize properly.
    A high learning rate can cause the map to change too drastically, not settling into a stable configuration,
    while a too low learning rate might prevent the map from organizing significantly during training. Similarly,
    an inappropriate neighborhood size might lead to either too local or too global organization.

    Parameters:
    - num_neurons (tuple): Dimensions of the SOM grid, e.g., (20, 20).
    - input_dim (int): Dimensionality of the input data.
    - learning_rate (float): Initial learning rate for the SOM training.
    - epochs (int): Number of iterations over the training dataset.

    Returns:
    - None

    Examples:
    ```
    # Creating an instance of SOM
    som = SOM(num_neurons=(20, 20), input_dim=40000, learning_rate=0.1, epochs=100)
    # Training the SOM with random data
    data = np.random.rand(200, 40000)
    som.train(data)
    ```

    Notes:
    - The implementation assumes that input data are normalized if necessary.
    - Proper parameter tuning (learning rate, epochs, neuron grid size) is crucial for effective maps.
    """

    def __init__(self, num_neurons=(20, 20), input_dim=40000, learning_rate=0.1, epochs=100):
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.random((num_neurons[0] * num_neurons[1], input_dim))
        self.history = []  # To store the loss at each epoch

    def train(self, data):
        step_print = max(1, self.epochs // 10)
        weight_changes = []
        for epoch in tqdm(range(self.epochs), desc="Training epochs"):
            epoch_errors = []
            prev_weights = np.copy(self.weights)
            for sample in data:
                bmu_idx = self.find_bmu(sample)
                self.update_weights(sample, bmu_idx, epoch)
                error = np.linalg.norm(sample - self.weights[bmu_idx])
                epoch_errors.append(error)
            epoch_loss = np.mean(epoch_errors)
            self.history.append(epoch_loss)
            weight_change = np.mean(np.linalg.norm(self.weights - prev_weights, axis=1))
            weight_changes.append(weight_change)
            if (epoch + 1) % step_print == 0:
                tqdm.write(f"Epoch {epoch+1}, Loss: {epoch_loss}, Weight Change: {weight_change}")

    def find_bmu(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=1)
        return np.argmin(distances)

    def update_weights(self, sample, bmu_idx, epoch):
        lr = self.learning_rate * np.exp(-epoch / self.epochs)
        self.weights[bmu_idx] += lr * (sample - self.weights[bmu_idx])

    def detect_anomalies(self, data, threshold=1.5):
        quantization_errors = np.array([np.linalg.norm(data[i] - self.weights[self.find_bmu(data[i])]) for i in range(len(data))])
        mean_error = np.mean(quantization_errors)
        std_error = np.std(quantization_errors)
        anomalies = np.where(quantization_errors > mean_error + threshold * std_error)[0]
        return anomalies

    def index_to_xy(self, index, width):
        return divmod(index, width)  # Returns (x, y)

    def get_activations(self, data):
        """Generate a matrix of activations for each neuron for given data."""
        activations = np.zeros((self.num_neurons[0], self.num_neurons[1], data.shape[0]))
        for i, sample in enumerate(data):
            bmu_idx = self.find_bmu(sample)
            x, y = divmod(bmu_idx, self.num_neurons[1])
            activations[x, y, i] = 1  # Mark the neuron as activated for this sample
        return activations

    def plot_u_matrix(self):
        """Plots the U-matrix of the SOM."""
        ux, uy = self.num_neurons
        u_matrix = np.zeros((ux, uy))
        iterator = np.nditer(u_matrix, flags=["multi_index"])
        while not iterator.finished:
            x, y = iterator.multi_index
            weight = self.weights[x * uy + y]
            count = 0
            dist = 0
            if x > 0:  # upper neighbor
                dist += np.linalg.norm(weight - self.weights[(x - 1) * uy + y])
                count += 1
            if x < ux - 1:  # lower neighbor
                dist += np.linalg.norm(weight - self.weights[(x + 1) * uy + y])
                count += 1
            if y > 0:  # left neighbor
                dist += np.linalg.norm(weight - self.weights[x * uy + (y - 1)])
                count += 1
            if y < uy - 1:  # right neighbor
                dist += np.linalg.norm(weight - self.weights[x * uy + (y + 1)])
                count += 1
            u_matrix[x, y] = dist / count
            iterator.iternext()

        plt.imshow(u_matrix, cmap="bone_r")
        plt.colorbar()
        plt.title("U-Matrix")
        plt.show()

    def plot_component_plane(self, component_idx):
        """Plots a component plane for the specified index of the SOM's weight vectors."""
        component_plane = self.weights[:, component_idx].reshape(self.num_neurons)
        plt.imshow(component_plane, cmap="viridis")
        plt.colorbar()
        plt.title(f"Component Plane for Feature {component_idx}")
        plt.show()


def plot_activations_on_image(image, activations, image_idx, anomalies: Union[List, list, None] = None):

    cmap = mcolors.LinearSegmentedColormap.from_list("ndwi", ["brown", "lightyellow", "blue"], N=256)

    # Create a normalized color bar for the NDWI range
    norm = MidpointNormalize(vmin=-1, vmax=1, midpoint=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    if anomalies:
        for anomaly in anomalies:
            plt.scatter(anomaly[1], anomaly[0], s=50, facecolor="none", edgecolor="r", marker="o")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap=cmap, norm=norm)
    plt.imshow(activations[:, :, image_idx], cmap="jet", alpha=0.5)  # Overlay activations
    plt.title("Image with SOM Activations")
    plt.colorbar()

    plt.show()


# Define a custom normalization class to emphasize differences
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
