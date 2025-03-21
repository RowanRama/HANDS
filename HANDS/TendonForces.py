import numpy as np
from elastica.typing import SystemType, RodType

# Normally NoForces would be included in the forcing module, but PyElastica requires it to be from the original elastica.external_forces module
from elastica.external_forces import NoForces
from numba import njit

class TendonForces(NoForces):
    """
    This class applies tendon forcing along the length of the rod.

        Attributes
        ----------
        vertebra_height: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        num_vertebrae: int
            Amount of vertebrae to be used in the system.
        vertebra_height_vector: numpy.ndarray
            1D (dim) numpy array. Describes the orientation and height in space of the vertebrae in the system.
        tension: float
            Tension applied to the tendon in the system.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        vertebra_weight_vector: numpy.ndarray
            1D (dim) numpy array. Vector which specifies the orientation and magnitude of the weight of the vertebrae (By default it is in the global -Z direction).
        vertebra_nodes: list
            1D (dim) list. Contains the node numbers of every node with vertebrae. The vertebrae are assumed to be uniformly spaced through the intervals specified by 
            first_vertebra_node and final_vertebra_node, with an amount equal to num_vertebrae.
        force_data: numpy.ndarray
            2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the nodes with vertebrae.
        

    """

    def __init__(self, vertebra_height, num_vertebrae, first_vertebra_node, final_vertebra_node, vertebra_mass, tension, vertebra_height_orientation, n_elements):
        """

        Parameters 
        ----------
        vertebra_height: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        num_vertebrae: int
            Amount of vertebrae to be used in the system.
        first_vertebra_node: int
            The first node to have a vertebra, from the base of the rod to the tip.
        final_vertebra_node: int
            The last node to have a vertebra, from the base of the rod to the tip.
        vertebra_mass: float
            Total mass of a single vertebra.
        tension: float
            Tension applied to the tendon in the system.
        vertebra_height_orientation: numpy.ndarray
            1D (dim) numpy array. Describes the orientatation of the vertebrae in the system.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        """
        super(TendonForces, self).__init__()

        # Initializing class attributes to be used in other methods
        self.vertebra_height = vertebra_height
        self.num_vertebrae = num_vertebrae
        self.vertebra_height_vector = vertebra_height_orientation * vertebra_height
        self.tension = tension
        self.n_elements = n_elements
        self.vertebra_weight_vector = np.array([0.0, 0.0, -vertebra_mass * 9.80665])

        # Creating vector containing the node numbers with the vertebras for this instance of TendonForces
        self.vertebra_nodes = []
        vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
        for i in range(num_vertebrae):
            self.vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

    def apply_forces(self, system: SystemType, time: np.float64 = 0.0):
        # The application of the force data is done outside of the @njit decorated function because self.force_data needs to be referenced in self.compute_torques()

        # Retrieves relative position unit norm vectors between each vertebra top (where the tendon contacts the vertebra)
        unit_norm_vector_array = self.get_rotations(np.array(system.position_collection), np.array(system.director_collection), np.array(self.vertebra_nodes), self.vertebra_height_vector)

        # Computes the forces in each vertebra
        self.force_data = self.compute_forces(self.tension, np.array(self.vertebra_nodes), unit_norm_vector_array)

        # Creating the force data set to apply to the rod
        apply_force = np.zeros((3,self.n_elements+1))

        # PyElastica handles forces in GLOBAL coord. system, so they are applied directly. Also, the vertebra weights are added to each vertebra
        for i in range (len(self.vertebra_nodes)):
            apply_force[:,self.vertebra_nodes[i]] = self.force_data[i] + self.vertebra_weight_vector

        # Applies forces to the rod
        system.external_forces += apply_force


    def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
        # The force_data set and vertebra_weight_vector are expressed in the global coordinate frame and must be changed to local reference frames for torque application
        # Creating the array which will contain the transformed force vectors
        transformed_force_data = np.zeros((len(self.vertebra_nodes), 3), dtype=np.float64)

        # Transforming the force vectors calculated in the compute_forces method from the global reference frame to the local reference frame
        for i in range(len(self.vertebra_nodes)):
            transformed_force_data[i] = system.director_collection[...,(self.vertebra_nodes[i]-1)] @ self.force_data[i]

        self.compute_torques(
            self.vertebra_height_vector, np.array(self.vertebra_nodes), transformed_force_data,
            self.n_elements, system.external_torques
        )


    @staticmethod
    @njit(cache=True)
    def get_rotations(position_collection, director_collection, vertebra_nodes, vertebra_height_vector):
        # Returns an array containing the unit norm vector which describes the orientation of each segment of tendon between vertebrae

        # Initializing unit_norm_vector_array to store the unit normed vectors that describe the global orientation of the forces in each vertebra
        unit_norm_vector_array = np.zeros((len(vertebra_nodes), 3), dtype=np.float64)

        for i in range(len(vertebra_nodes)+1):
            # There is a +1 in the for loop to account for the force between the first vertebra and the fixed node

            # If statement, used for the case when i = 0 and thus there is no vertebra before this one, same for the final vertebra (no vertebra after that one)
            if i==0:
                current_vertebra = 0
                next_vertebra = vertebra_nodes[i]
            elif i==len(vertebra_nodes):
                current_vertebra = vertebra_nodes[i-1]
                next_vertebra = vertebra_nodes[i-1]
            else:
                current_vertebra = vertebra_nodes[i-1]
                next_vertebra = vertebra_nodes[i]

            # Setting up values to be used iteratively
            x_current = position_collection[0, current_vertebra]
            y_current = position_collection[1, current_vertebra]
            z_current = position_collection[2, current_vertebra]

            x_next = position_collection[0, next_vertebra]
            y_next = position_collection[1, next_vertebra]
            z_next = position_collection[2, next_vertebra]

            current_rotation_matrix = director_collection[...,current_vertebra]
            next_rotation_matrix = director_collection[...,next_vertebra]

            current_node = np.array([x_current, y_current, z_current])
            next_node = np.array([x_next, y_next, z_next])

            # Calculating relative position vector between vertebrae, considering the vertebra height
            # Continguous arrays to help with computation speed
            delta_vector = (next_node + np.ascontiguousarray(next_rotation_matrix.T) @ np.ascontiguousarray(vertebra_height_vector)) - (current_node + np.ascontiguousarray(current_rotation_matrix.T) @ np.ascontiguousarray(vertebra_height_vector))

            # Calculating the unit-normed vector based on the differences calculated in the previous step
            delta_vector_norm = np.linalg.norm(delta_vector)
            unit_norm_delta_vector = delta_vector / delta_vector_norm

            # This if statement is to stop unit_norm_delta_vector from becoming a 'nan'
            if i==len(vertebra_nodes):
                unit_norm_delta_vector = np.zeros(3)

            # Storing the unit normed vector to be later used in the compute_forces method
            unit_norm_vector_array[i] = unit_norm_delta_vector

        return unit_norm_vector_array

    @staticmethod
    @njit(cache=True)
    def compute_forces(tension, vertebra_nodes, unit_norm_vector_array):

        # Creating array to store forces in vertebrae
        force_data = np.zeros((len(vertebra_nodes), 3), dtype=np.float64)

        for i in range(len(vertebra_nodes)):
            # This for loop multiplies the unit normed vectors calculated previously, with the tension of the tendon, thus creating the force vector for each vertebra
            # Contiguous array to increase speed in njit decorator
            force_current_prev = unit_norm_vector_array[i] * -tension
            force_current_next = unit_norm_vector_array[i+1] * tension

            # Summing the components of both force vectors to get the final force vector, which is then stored for use in the apply_forces and compute_torques methods
            force_data[i] = force_current_prev + force_current_next

        return force_data


    @staticmethod
    @njit(cache=True)
    def compute_torques(vertebra_height_vector, vertebra_nodes, transformed_force_data, n_elements, external_torques):

        # Creating torque data set for storage
        torque_data = np.zeros((len(vertebra_nodes), 3),dtype=np.float64)

        # Goes through vertebra nodes to calculate torques for them
        for i in range(len(vertebra_nodes)):

            # Cross product between the vertebra height vector and the local force vector due to the tendons, to obtain the tendon torque for that vertebra
            torque_vector = np.cross(vertebra_height_vector, transformed_force_data[i])

            # Sum of the vectors, and storage into the torque_data array
            torque_data[i] = torque_vector

        # Appending the computed torque vector to the final torque data set
        apply_torque = np.zeros((3,n_elements+1))

        k = 0
        for i in range(n_elements):
            if i in vertebra_nodes:
                apply_torque[:,i] = torque_data[k]
                k += 1
        apply_torque = apply_torque[:,1:]

        # Applying the torque data set to the rod (torque on the final vertebra)
        external_torques += apply_torque
