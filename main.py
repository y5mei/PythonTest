import tensorflow as tf
import numpy as np
import copy

import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    # To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        # Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 5
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
        self._centroid_grid_detial = [[] for i in range(self._n_iterations)]

        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            # Randomly initialized weightage vectors for all neurons,
            # stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random.uniform(
                [m * n, dim], minval=0, maxval=1))

            # Matrix of size [m*n, 2] for SOM grid locations
            # of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they
            # will be fed in during training

            # The training vector
            self._vect_input = tf.compat.v1.placeholder("float", [dim])
            # Iteration number
            self._iter_input = tf.compat.v1.placeholder("float")

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            # To compute the Best Matching Unit given a vector
            # Basically calculates the Euclidean distance between every
            # neuron's weightage vector and the input, and returns the
            # index of the neuron which gives the least value
            bmu_index = tf.argmin(input=tf.sqrt(tf.reduce_sum(
                input_tensor=tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m * n)])), 2), axis=1)),
                axis=0)

            # This will extract the location of the BMU based on the BMU's
            # index
            slice_input = tf.pad(tensor=tf.reshape(bmu_index, [1]),
                                 paddings=np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])

            # To compute the alpha and sigma values based on iteration
            # number
            learning_rate_op = tf.subtract(1.0, tf.compat.v1.div(self._iter_input,
                                                                 self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            # Construct the op that will generate a vector with learning
            # rates for all neurons, based on iteration number and location
            # wrt BMU.
            bmu_distance_squares = tf.reduce_sum(input_tensor=tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m * n)])), 2), axis=1)
            neighbourhood_func = tf.exp(tf.negative(tf.compat.v1.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update
            # the weightage vectors of all neurons based on a particular
            # input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                for i in range(m * n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m * n)]),
                            self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.compat.v1.assign(self._weightage_vects,
                                                    new_weightages_op)

            ##INITIALIZE SESSION
            self._sess = tf.compat.v1.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.compat.v1.initialize_all_variables()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train_detial(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        # Prepare a list of 2D list to store the result for each step
        import copy
        centroid_grid_detial = [[] for i in range(self._n_iterations)]

        # Training iterations
        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op, feed_dict={self._vect_input: input_vect, self._iter_input: iter_no})
            # Store

            # Store a centroid grid for easy retrieval later on
            centroid_grid = [[] for i in range(self._m)]
            self._weightages = list(self._sess.run(self._weightage_vects))
            self._locations = list(self._sess.run(self._location_vects))
            for i, loc in enumerate(self._locations):
                centroid_grid[loc[0]].append(self._weightages[i])
            self._centroid_grid = centroid_grid

            # Sotre this centroid grid to the grid list
            centroid_grid_detial[iter_no] = copy.deepcopy(centroid_grid)

        self._centroid_grid_detial = centroid_grid_detial
        self._trained = True

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        # Training iterations
        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def get_centroids_detial(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid_detial

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return

    def get_ephoch(self):
        return self._n_iterations

    # input a trained SOM Map, and a RGB Color
    # return a list in length of m*n, sorted by distance close to the RGB color
    # in the format of [[dist_to_the_RGB, row, col,[RGBColor]]]
    def sorted_bmu_list(self, traied_map, input_color):
        result = []
        for i in range(self._m):
            for j in range(self._n):
                trained_color = traied_map[i][j]
                diff = sum([(a_i - b_i) * (a_i - b_i) for a_i, b_i in zip(trained_color, input_color)])
                element = [diff, i, j, copy.deepcopy(traied_map[i][j])]
                result.append(element)

        r = sorted(result, key=lambda x: x[0])
        return r

    # Compare two RGB colors, return True if they are the same in int 255 base
    def is_same_color(colorA, colorB):
        for i in range(len(colorA)):
            if int(round(colorA[i] * 255)) != int(round(colorB[i] * 255)):
                return False
        return True

    # return True if two coordinates are neighbor
    def is_neighbor_neurol(x1, y1, x2, y2):
        if (x1 == x2) and (y1 == y2):
            return False
        dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        return dist <= 2

    # return true is the second best match unit is a neighbor of the bmu:
    def second_bmu_neightbor_of_bmu(sorted_list):
        result = False
        mybmu = sorted_list[0]
        list_of_second_bmu = SOM.sorted_second_bmu_list(sorted_list)

        for e in list_of_second_bmu:
            if SOM.is_neighbor_neurol(mybmu[1], mybmu[2], e[1], e[2]):
                return True
        return False

    # return the list of the second best match units
    # if there are multiple second bmu in the [0,255] scale, return them as a list together
    def sorted_second_bmu_list(sorted_list):
        result = []
        current_bmu_color = sorted_list[0][-1]
        # return current_bmu_color

        # if there are more than two bmu, return all the bmu:
        if SOM.is_same_color(sorted_list[0][-1], sorted_list[1][-1]):
            for e in sorted_list:
                if SOM.is_same_color(sorted_list[0][-1], e[-1]):
                    result.append(e)
                else:
                    return result

        else:
            # if there are only one bmu, return all the neurons equals the second bmu:
            for e in sorted_list[1:]:
                if SOM.is_same_color(sorted_list[1][-1], e[-1]):
                    result.append(e)
                else:
                    return result


# For plotting the images
from matplotlib import pyplot as plt

# Training inputs for RGBcolors
colors = np.array(
    [ [1.0,0.647,0.0],
     [0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.5],
     [0.125, 0.529, 1.0],

     [0.33, 0.4, 0.67],
     [0.6, 0.5, 1.0],
     [0., 1., 0.],
     [1., 0., 0.],

     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],

     [.33, .33, .33],
     [.5, .5, .5],
     [.66, .66, .66]])
color_names = \
    ['orange','black', 'blue', 'darkblue',
     'skyblue','greyblue', 'lilac', 'green',
     'red','cyan', 'violet', 'yellow',
     'white','darkgrey', 'mediumgrey', 'lightgrey']

# Train a 50x50 SOM with 400 iterations
# 50 行，50 列， 3维GRB 输入，训练5 epoch
num_epoch = 3
som = SOM(15, 15, 3, num_epoch)
# som.train(colors)
# The Training of a SOM is basically a map from a 3D space (R,G,B)
# to a 2-d mainfold topo-space via the similarity of the input 3D vector
# to each cell in the SOM grid
som.train_detial(colors)

# Get output grid
# check RGB color here: https://www.easyrgb.com/en/convert.php#inputFORM
image_grid = som.get_centroids()
image_grid_detial = som.get_centroids_detial()

print(image_grid_detial[num_epoch-1])
print("#################################")
print(image_grid[0][0])
print(image_grid[0][1])
print(image_grid[1][0])
print(image_grid[1][1])



# Plot
plt.imshow(image_grid)
titleString = 'Color SOM \n' + 'Epoch =' + str(som.get_ephoch())
plt.title(titleString)

# Map colours to their closest neurons
mapped = som.map_vects(colors)
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()



