import numpy as np
import random
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation as R
import numpy as np
import random
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns
import json


class Peptide:
    default_parameters = {
        "l": 0.38,
        "k_bond": 4184,
        "k_angle": 4.184,
        "theta": 3.14,
    }

    def __init__(
        self, sequence, parameter_file="params.json", starting_conformation=None
    ):
        n_beads = len(sequence)

        self.parameters = json.load(open(parameter_file, "r"))

        self.sequence = sequence

        self.n_beads = n_beads
        self.conformations = []

        if starting_conformation is None:
            conformation = [[3.834, 2.997, 2.635]]
        else:
            conformation = starting_conformation

        while len(conformation) < n_beads:
            direction = np.array([0, 0, 1.0])
            # add some noise to the direction
            direction += np.random.normal(0, 0.5, 3)

            direction /= np.linalg.norm(direction)

            if (
                sequence[len(conformation) - 1 : len(conformation) + 1]
                in self.parameters
            ):
                bond_length = self.parameters[
                    sequence[len(conformation) - 1 : len(conformation) + 1]
                ]["l"]
            else:
                bond_length = self.default_parameters["l"]
            position = conformation[-1] + (direction * bond_length)
            conformation.append(list(position))

        self.conformations.append(conformation)

        self.compute_distances()

        self.kappa = 1

        self.set_sigma()
        self.set_epsilon()
        self.set_A0()
        self.set_A()

        self._bonds = None
        self._angles = None
        self._dihedrals = None
        self._pairwise_distances = None

    @property
    def conformation(self):
        return self.conformations[-1] if len(self.conformations) > 0 else None

    def add_conformation(self):
        if self.conformation is not None:
            self.conformations.append(deepcopy(self.conformation))

    def save_pdb(self, filename, interval=1):
        with open(filename, "w") as f:
            for model_index, positions in enumerate(self.conformations, start=1):
                if model_index % interval != 0:
                    continue
                f.write(f"MODEL     {model_index}\n")
                for atom_index, position in enumerate(positions, start=1):
                    x, y, z = position
                    f.write(
                        f"ATOM  {atom_index:>5}  CA  {self.sequence[atom_index-1]}   A{atom_index:>4}    {x*10:>8.3f}{y*10:>8.3f}{z*10:>8.3f}  1.00  0.00           C  \n"
                    )
                f.write("ENDMDL\n")

    def get_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def get_angles_matrix(self, a, b, c):
        BA = a - b
        BC = c - b

        dot_product = np.sum(BA * BC, axis=1)
        magnitude_BA = np.linalg.norm(BA, axis=1)
        magnitude_BC = np.linalg.norm(BC, axis=1)

        cosine_angle = dot_product / (magnitude_BA * magnitude_BC)
        angles = np.arccos(
            np.clip(cosine_angle, -1.0, 1.0)
        )  # Clip to handle potential numerical errors

        # If you want to consider the direction of the rotation like in the original function:
        cross_product_z = BA[:, 0] * BC[:, 1] - BA[:, 1] * BC[:, 0]
        angles[cross_product_z < 0] = 2 * np.pi - angles[cross_product_z < 0]

        return angles

    def get_dihedral(self, a, b, c, d):
        b1 = np.array(b) - np.array(a)
        b2 = np.array(c) - np.array(b)
        b3 = np.array(d) - np.array(c)

        c1 = np.cross(b2, b3)
        c2 = np.cross(b1, b2)

        p1 = (b1 * c1).sum(-1)
        p1 *= (b2 * b2).sum(-1) ** 0.5
        p2 = (c1 * c2).sum(-1)

        return np.arctan2(p1, p2)

    @property
    def bonds(self):
        if self._bonds is None:
            self._bonds = np.linalg.norm(np.diff(self.conformation, axis=0), axis=1)
        return self._bonds

    @property
    def angles(self):
        if self._angles is None:
            points = np.array(self.conformation)
            self._angles = self.get_angles_matrix(points[:-2], points[1:-1], points[2:])
        return self._angles

    @property
    def dihedrals(self):
        if self._dihedrals is None:
            self._dihedrals = [
                self.get_dihedral(
                    self.conformation[i],
                    self.conformation[i + 1],
                    self.conformation[i + 2],
                    self.conformation[i + 3],
                )
                for i in range(self.n_beads - 3)
            ]
        return self._dihedrals

    @property
    def pairwise_distances(self):
        if self._pairwise_distances is None:
            distances = self.distances_matrix.astype(np.float64)
            distances = np.triu(
                distances, k=3
            )  # gets upper triangle with k=3 to skip diagonal and adjacent elements
            self._pairwise_distances = distances[
                distances != 0
            ]  # filters out zero values which corresponds to the masked elements
        return self._pairwise_distances

    def bond_energy(self, bond_lengths, indices):
        sequences = [self.sequence[i : i + 2] for i in indices]
        kappas = np.array(
            [
                self.parameters[seq]["l"]
                if seq in self.parameters
                else self.default_parameters["l"]
                for seq in sequences
            ]
        )
        k_bonds = np.array(
            [
                self.parameters[seq]["k_bond"]
                if seq in self.parameters
                else self.default_parameters["k_bond"]
                for seq in sequences
            ]
        )

        length_diffs = bond_lengths - kappas
        return 0.5 * k_bonds * np.power(length_diffs, 2)

    def angle_energy(self, thetas, indices):
        sequences = [self.sequence[i : i + 3] for i in indices]
        k_angles = np.array(
            [
                self.parameters[seq]["k_angle"]
                if seq in self.parameters
                else self.default_parameters["k_angle"]
                for seq in sequences
            ]
        )
        target_thetas = np.array(
            [
                self.parameters[seq]["theta"]
                if seq in self.parameters
                else self.default_parameters["theta"]
                for seq in sequences
            ]
        )

        theta_diffs = thetas - target_thetas
        return 0.5 * k_angles * np.power(theta_diffs, 2)

    def calculate_short_range_energies(self):
        sigma_over_distances = np.divide(
            self.sigma[: len(self.pairwise_distances)], self.pairwise_distances
        )
        sigma_over_distances_5 = sigma_over_distances**5
        energies = (
            4
            * self.epsilon[: len(self.pairwise_distances)]
            * (sigma_over_distances_5 * (sigma_over_distances_5 - 1))
        )
        return np.sum(energies)

    def calculate_bond_energies(self):
        valid_bonds_indices = np.where(~np.isnan(self.bonds))[
            0
        ]  # Indices of bonds that are not NaN
        valid_bonds = self.bonds[valid_bonds_indices]

        energies = self.bond_energy(valid_bonds, valid_bonds_indices)
        return np.sum(energies)

    def calculate_angle_energies(self):
        valid_angles_indices = np.where(~np.isnan(self.angles))[
            0
        ]  # Indices of angles that are not NaN
        valid_angles = self.angles[valid_angles_indices]

        energies = self.angle_energy(valid_angles, valid_angles_indices)
        return np.sum(energies)

    def calculate_electrostatic_energy(self):
        A_plus_A0_over_distances = np.divide(
            (self.A + self.A0)[: len(self.pairwise_distances)], self.pairwise_distances
        )
        return np.sum(
            A_plus_A0_over_distances * np.exp(-self.pairwise_distances / self.kappa)
        )

    def calculate_energy(
        self,
        bond=True,
        angle=True,
        short_range=True,
        electrostatic=True,
    ):
        self.compute_distances()
        energies = []

        if bond:
            energies.append(self.calculate_bond_energies())
        if angle:
            energies.append(self.calculate_angle_energies())
        if short_range:
            energies.append(self.calculate_short_range_energies())
        if electrostatic:
            energies.append(self.calculate_electrostatic_energy())

        return np.round(
            np.array(energies),
            3,
        )

    def stretch_bond(self, bead_index, distance):
        conformation = np.array(self.conformations[-1])

        # Get the vector between the two beads
        vector = conformation[bead_index + 1] - conformation[bead_index]
        vector /= np.linalg.norm(vector)

        # Translate the subsequent beads
        for i in range(bead_index + 1, self.n_beads):
            conformation[i] += distance * vector

        self.conformations.append(conformation)

        self._dihedrals = None
        self._pairwise_distances = None
        self._angles = None
        self._bonds = None

    def rotate_first_bond(self, angle):
        conformation = np.array(self.conformations[-1])

        rotation_axis = np.array(conformation[1]) - np.array(conformation[0])

        rotation_axis /= np.linalg.norm(rotation_axis)

        rotation_matrix = R.from_rotvec(rotation_axis * angle).as_matrix()

        pivot = np.array(conformation[0])
        for i in range(1, self.n_beads):
            conformation[i] = list(
                np.dot(rotation_matrix, np.array(conformation[i]) - pivot) + pivot
            )

        self.conformations.append(conformation)

        self._dihedrals = None
        self._pairwise_distances = None
        self._angles = None
        self._bonds = None

    def rotate_bond(self, bead_index, angle):
        conformation = np.array(self.conformations[-1])

        # Define the rotation axis as the central bond of the dihedral
        rotation_axis = np.array(conformation[bead_index + 2]) - np.array(
            conformation[bead_index + 1]
        )
        rotation_axis /= np.linalg.norm(rotation_axis)

        # Get the rotation matrix
        rotation_matrix = R.from_rotvec(rotation_axis * angle).as_matrix()

        pivot = np.array(conformation[bead_index + 1])
        for i in range(bead_index + 2, self.n_beads):
            # Update the position of the subsequent beads
            conformation[i] = list(
                np.dot(rotation_matrix, np.array(conformation[i]) - pivot) + pivot
            )

        self.conformations.append(conformation)

        self._dihedrals = None
        self._pairwise_distances = None
        self._angles = None
        self._bonds = None

    def random_position(self, bead_index, angle):
        conformation = np.array(self.conformations[-1])

        # Rotate the bead around a random axis
        rotation_axis = np.random.rand(3)
        rotation_axis /= np.linalg.norm(rotation_axis)

        # Get the rotation matrix
        rotation_matrix = R.from_rotvec(rotation_axis * angle).as_matrix()

        pivot = np.array(conformation[bead_index])
        for i in range(bead_index + 1, self.n_beads):
            # Update the position of the subsequent beads
            conformation[i] = list(
                np.dot(rotation_matrix, np.array(conformation[i]) - pivot) + pivot
            )

        self.conformations.append(conformation)

        self._dihedrals = None
        self._pairwise_distances = None
        self._angles = None
        self._bonds = None

    def compute_distances(self):
        conformation = np.array(self.conformation)

        all_distances = pdist(conformation, "euclidean")

        self.distances_matrix = squareform(all_distances)

    def set_sigma(self):
        a = []
        for i in range(self.n_beads):
            row = []
            for j in range(self.n_beads):
                row.append(
                    (
                        self.parameters[self.sequence[i]]["r"]
                        + self.parameters[self.sequence[j]]["r"]
                    )
                    / 2
                )
            a.append(row)
        a = np.array(a).astype(dtype=np.float64)
        i, j = np.indices(a.shape)

        a[(abs(i - j) <= 1)] = np.nan

        self.sigma = a[~np.isnan(a)]

    def set_epsilon(self):
        a = []
        for i in range(self.n_beads):
            row = []
            for j in range(self.n_beads):
                row.append(
                    np.sqrt(
                        self.parameters[self.sequence[i]]["epsilon"]
                        * self.parameters[self.sequence[j]]["epsilon"]
                    )
                )
            a.append(row)
        a = np.array(a).astype(dtype=np.float64)
        i, j = np.indices(a.shape)

        a[(abs(i - j) <= 1)] = np.nan

        self.epsilon = a[~np.isnan(a)]

    def set_A0(self):
        a = []
        for i in range(self.n_beads):
            row = []
            for j in range(self.n_beads):
                row.append(
                    self.parameters[self.sequence[i]]["A0"]
                    + self.parameters[self.sequence[j]]["A0"]
                )
            a.append(row)
        a = np.array(a).astype(dtype=np.float64)
        i, j = np.indices(a.shape)

        a[(abs(i - j) <= 1)] = np.nan

        self.A0 = a[~np.isnan(a)]

    def set_A(self):
        a = []
        for i in range(self.n_beads):
            row = []
            for j in range(self.n_beads):
                row.append(
                    self.parameters[self.sequence[i]]["A"]
                    * self.parameters[self.sequence[j]]["A"]
                )
            a.append(row)
        a = np.array(a).astype(dtype=np.float64)
        i, j = np.indices(a.shape)

        a[(abs(i - j) <= 1)] = np.nan

        self.A = a[~np.isnan(a)]


class MC:
    def __init__(self, peptide: Peptide):
        """Monte Carlo simulation of a peptide chain.

        Args:
            peptide (Peptide): Peptide to simulate."""

        self.peptide = peptide

    def run(
        self,
        n_steps,
        bond_energy=True,
        angle_energy=True,
        short_range_energy=True,
        electrostatic_energy=True,
        stretch_bond=True,
        rotate_bead=True,
        rotate_dihedral=True,
        rotate_first=True,
        plot=True,
        save_file=False,
        c_ratio=1,
        T=300,
    ):
        """Run the Monte Carlo simulation.

        Args:
            n_steps (int): Number of steps to run the simulation for.
            bond_energy (bool, optional): Whether to include bond energy in the simulation. Defaults to True.
            angle_energy (bool, optional): Whether to include angle energy in the simulation. Defaults to True.
            short_range_energy (bool, optional): Whether to include short range energy in the simulation. Defaults to True.
            electrostatic_energy (bool, optional): Whether to include electrostatic energy in the simulation. Defaults to True.
            stretch_bond (bool, optional): Whether to include stretching bonds in the simulation. Defaults to True.
            rotate_bead (bool, optional): Whether to include rotating beads in the simulation. Defaults to True.
            rotate_dihedral (bool, optional): Whether to include rotating dihedrals in the simulation. Defaults to True.
            rotate_first (bool, optional): Whether to include rotating the first bond in the simulation. Defaults to True.
            plot (bool, optional): Whether to plot the simulation. Defaults to True.
            save_file (bool, optional): Whether to save the simulation as a .pdb file. Defaults to False.
            c_ratio (int, optional): Ratio of the chain length to the Kuhn length. Defaults to 1.
            T (int, optional): Temperature of the simulation. Defaults to 300.
        """

        if save_file:
            if not save_file.endswith(".pdb"):
                raise ValueError("Save file must be a .pdb file")

        energy = self.peptide.calculate_energy(
            bond=bond_energy,
            angle=angle_energy,
            short_range=short_range_energy,
            electrostatic=electrostatic_energy,
        )

        self.distances = []
        self.termini = []
        self.energies = []
        self.moves = []

        for i in range(n_steps):
            options = []
            if stretch_bond:
                options.append("stretch")
            if rotate_bead:
                options.append("rotate")
            if rotate_dihedral:
                options.append("dihedral")
            if rotate_first:
                options.append("rotate_first")
            options.append("rotate_first")

            while True:
                choice = random.choice(options)

                if choice == "stretch":
                    distance = np.random.uniform(-0.1, 0.1)
                    bead = np.random.randint(self.peptide.n_beads - 1)
                    self.peptide.stretch_bond(bead, distance)
                elif choice == "rotate":
                    angle = np.random.uniform(-np.radians(180), np.radians(180))
                    bead = np.random.randint(self.peptide.n_beads - 1)
                    self.peptide.random_position(bead, angle)
                elif choice == "dihedral":
                    angle = np.random.uniform(-np.radians(20), np.radians(20))
                    bead = np.random.randint(self.peptide.n_beads - 3)
                    self.peptide.rotate_bond(bead, angle)
                elif choice == "rotate_first":
                    angle = np.random.uniform(-np.radians(5), np.radians(5))
                    self.peptide.rotate_first_bond(angle)

                else:
                    print("No available moves")
                    return

                break

            new_energy = self.peptide.calculate_energy(
                bond=bond_energy,
                angle=angle_energy,
                short_range=short_range_energy,
                electrostatic=electrostatic_energy,
            )

            random_number = random.random()

            if new_energy.sum() == np.inf:
                accept = False
            elif new_energy.sum() < energy.sum():
                accept = True
            elif random_number < np.exp(
                -(new_energy.sum() - energy.sum()) / (T * 0.008314462618)
            ):
                accept = True
            else:
                accept = False

            if accept:
                energy = new_energy
                self.distances.append(
                    10
                    * (
                        np.linalg.norm(
                            np.array(self.peptide.conformation[0])
                            - np.array(self.peptide.conformation[-1])
                        )
                    )
                )
                self.termini.append(np.array(self.peptide.conformation[-1]) * 10)
                self.energies.append(energy.sum())
            else:
                self.peptide.conformations.pop()

            self.moves.append((choice, accept))

            if i % 1000 == 0 and len(self.distances) > 1 and plot:
                clear_output(wait=True)
                plt.close()  # Clear the previous plot

                fig, ax = plt.subplots(1, 2, figsize=(12, 4))

                sns.kdeplot(self.distances, ax=ax[0], color="k")

                # plot the mean as a vertical line
                ax[0].axvline(
                    np.sqrt(np.mean(np.array(self.distances) ** 2)),
                    color="k",
                    linestyle="dashed",
                    linewidth=1,
                )

                # plot end to end probability distribution of gaussian chain
                # Range of end-to-end distances to consider
                r = np.linspace(np.min(self.distances), np.max(self.distances), 100)

                # Calculate the end-to-end distance probability distribution
                # using GC flory model
                b = 3.8
                n = self.peptide.n_beads - 1
                r_0 = c_ratio * n * (b**2)

                P_r = (
                    4
                    * np.pi
                    * r**2
                    * (3 / (2 * np.pi * r_0)) ** (3 / 2)
                    * np.exp(-3 * r**2 / (2 * r_0))
                )

                ax[0].plot(r, P_r, color="r", linestyle="dashed", linewidth=1)

                energy_str = ""

                if bond_energy:
                    energy_str += f"Bond: {np.round(energy[0], 1)} "
                if angle_energy:
                    energy_str += f"Angle: {np.round(energy[1], 1)} "
                if short_range_energy:
                    energy_str += f"Short range: {np.round(energy[2], 1)} "
                if electrostatic_energy:
                    energy_str += f"Electrostatic: {np.round(energy[3], 1)} "

                ax[0].set_title(
                    f"Step {i} - Energy: {energy_str} - Total: "
                    + f"{np.round(energy.sum())} "
                )
                ax[0].set_xlabel("Distance")
                ax[0].set_ylabel("Frequency")

                ax[0].axvline(
                    np.sqrt(c_ratio * n * (b**2)),
                    color="r",
                    linestyle="dashed",
                    linewidth=1,
                )

                ax[1].plot(self.energies[int(len(self.energies) / 10) :])
                ax[1].set_ylabel("Energy")

                plt.show()  # Display the plot
            if i % 1000 == 0 and len(self.distances) > 1 and save_file:
                if save_file:
                    self.peptide.save_pdb(save_file, interval=100)
