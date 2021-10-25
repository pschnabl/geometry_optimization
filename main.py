import sys
import os
import time
import numpy as np
from random import choices, randint, randrange, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyscf import gto, scf


# Function to plot molecule.
def plot_molecule(zmat, title='default', ret_fig=False):
    mol = gto.M(atom=zmat)
    atomnames = mol.elements
    xyzarray = mol.atom_coords('Angstrom')
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig=fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)  # plt.figure()
    ax.set_title(label=title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    colors = {'C': 'black', 'H': 'blue', 'O': 'red', 'F': 'green'}
    for i in range(len(atomnames)):
        ax.scatter(xyzarray[i, 0], xyzarray[i, 1], xyzarray[i, 2],
                   color=colors[atomnames[i]], s=100)
        ax.text(xyzarray[i, 0], xyzarray[i, 1], xyzarray[i, 2],
                '%s' % (str(i + 1)), size=10, zorder=1, color='k')

    if ret_fig:
        return fig


def changetorsionangle(zmat, newtorsionangles, atomindices=None):
    # takes zmat as input (needs to be a triple quoted string with the first line containing only the first C)
    # returns new zmat with torsion angles changes according to the given list newangles (list with 5 strings seperated
    # by comma, each entry being a string holding the value of a torsion angle e.g. ['120', '180', '180', '180', '60'] )
    if atomindices is None:
        atomindices = [11, 13, 15, 18, 19]

    # initialization
    zmatlist = []
    oldangles = []
    i = 0
    for ind, line in enumerate(zmat.splitlines()):
        # print(line)
        line_elements = line.split()  # list containing each element of a line sperated by ,
        # oldangles.append(line_elements[-1])  # save old angles, may be needed later for genetic algorithm?
        if ind in atomindices:
            line_elements[-1] = str(newtorsionangles[i])  # change H torsion angle
            i = i + 1
        zmatlist.append(line_elements)
    newzmat = ''
    for line in zmatlist:
        for char in line:
            newzmat = newzmat + char + " "
        newzmat = newzmat + '\n'
    return newzmat


def energy(zmat):
    mol = gto.M(atom=zmat, basis='sto-3g')
    return scf.RHF(mol).kernel()


def energyfromtorsionangles(newtorsionangles, zmat=None, atomindices=None):
    if atomindices is None:
        atomindices = [11, 13, 15, 18, 19]
    if zmat is None:
        zmat = """C		
    C	1	1.400000	
    C	2	1.400000	1	120.000000		
    C	3	1.400000	2	120.000000		1	0.000000
    C	4	1.400000	3	120.000000		2	0.000000
    C	5	1.400000	4	120.000000		3	0.000000
    C	6	1.340000	5	120.000000		4	180.000000
    C	1	1.340000	2	120.000000		3	180.000000
    C	8	1.340000	1	120.000000		2	180.000000
    C	7	1.340000	6	120.000000		5	180.000000
    O	7	1.240000	6	120.000000		5	0.000000
    H	11	0.890000	7	120.000000		6	180.000000
    O   5	1.240000	4	120.000000		3	180.000000
    H	13	0.890000	5	120.000000		6	180.000000
    O	3	1.240000	4	120.000000		5	180.000000
    H	15	0.890000	3	120.000000		4	180.000000
    O	8	1.240000	1	120.000000		6	180.000000
    O	9	1.240000	8	120.000000		1	180.000000
    H	17	0.890000	8	120.000000		9	180.000000
    H	18	0.890000	9	120.000000		8	180.000000
    H	10	0.990000	7	120.000000		6	180.000000
    F	2	1.390000	3	109.471000		4	180.000000
    F	4	1.390000	3	120.000000		2	180.000000"""
    new_zmat = changetorsionangle(zmat, newtorsionangles, atomindices)
    return energy(new_zmat)


def generate_genome(possible_angles, genome_length):
    # generates a genome of different angles corresponding to the torsion of OH-groups e.g. [120,60,120,0,180]
    return choices(possible_angles, k=genome_length)


def generate_population(size, possible_angles, genome_length):
    # creates different sets of torsion-angles corresponding to the torsion of OH-groups
    return [generate_genome(possible_angles, genome_length) for _ in range(size)]


def single_point_crossover(a, b):
    # cut genome at a random point and create two new genomes that are combinations of the cuts of a and b.
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome, num, probability, possible_angles):
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else choices(possible_angles)[0]
    return genome


def population_fitness(population, z_mat=None):
    return [energyfromtorsionangles(genome, z_mat) for genome in population]


def pair_selection(sorted_population, sorted_fitness):
    # ranked pair selection
    rank = [sorted_fitness.index(fitness) + 1 for fitness in sorted_fitness]
    rank.reverse()  # lowest energy has the highest rank
    return choices(
        population=sorted_population,
        weights=rank,
        k=2
    )


def ga(population_size, num_generations, genome_length, possible_angles, mutation_prob, z_matrix=None):
    best_genomes = []
    best_energies = []
    avg_energies = []
    all_energies = []
    population = generate_population(size=population_size, possible_angles=possible_angles, genome_length=genome_length)
    initial_pop = population
    for gen in range(num_generations):
        genome_fitnesses = population_fitness(population, z_matrix)
        avg_pop_fitness = sum([genomefitness / len(genome_fitnesses) for genomefitness in genome_fitnesses])
        print('Generation: ', gen, '    Avg. Population Fitness: ', avg_pop_fitness)
        fitness_and_pop_sorted = sorted(zip(genome_fitnesses, population))  # [(-942.5, [240, 0, 180, 300, 120]), ... ]
        sorted_fitness = [elem[0] for elem in fitness_and_pop_sorted]
        sorted_population = [elem[1] for elem in fitness_and_pop_sorted]

        best_genomes.append(sorted_population[0])
        best_energies.append(sorted_fitness[0])
        avg_energies.append(avg_pop_fitness)
        for genomefitness in genome_fitnesses:
            all_energies.append(genomefitness)

        print('Best Genome: ', sorted_population[0], '  Energy: ', sorted_fitness[0])
        print("----------")

        next_generation = sorted_population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = pair_selection(sorted_population, sorted_fitness)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a, num=1, probability=mutation_prob, possible_angles=possible_angles)
            offspring_b = mutation(offspring_b, num=1, probability=mutation_prob, possible_angles=possible_angles)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
    final_pop = population
    return best_genomes[-1], best_energies, avg_energies, all_energies, initial_pop, final_pop


def plot_genome(genome, zmat):
    zmat = changetorsionangle(zmat, genome)
    plot_molecule(zmat)
    # plt.show()


def plotfunc(best_energies, avg_energies, all_energies, popsize, mutprob, dirname, savefig=True):
    dirname = dirname + '/'

    plt.figure(figsize=(8, 6))
    plt.plot(avg_energies, label='average energy')
    plt.plot(best_energies, label='minimum energy')
    plt.legend(fontsize=12)
    plt.title('Average and minimum energy: \n'
              + ' population size = %i,' % popsize + ' mutation rate = %1.2f' % mutprob +
              '\n smallest energy: %1.5f' % best_energies[-1], fontsize=12)
    plt.xlabel('generation', fontsize=15)
    plt.ylabel('E', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if savefig is True:
        plt.savefig(dirname + 'Emin_Eavg_popsize_' + str(popsize) + '_mutrate_' + str(mutprob) + '.png', dpi=100)
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(all_energies, '.', label='all energies')
    plt.legend(fontsize=12)
    plt.title('Energy spread:' + ' population size = %i,' % popsize + ' mutation rate = %1.2f' % mutprob)
    plt.xlabel('Index', fontsize=15)
    plt.ylabel('E', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if savefig is True:
        plt.savefig(dirname + 'allEnergies_popsize_' + str(popsize) + '_mutrate_' + str(mutprob) + '.png', dpi=100)
    # plt.show()


# main script

zmat = """C		
    C	1	1.400000	
    C	2	1.400000	1	120.000000		
    C	3	1.400000	2	120.000000		1	0.000000
    C	4	1.400000	3	120.000000		2	0.000000
    C	5	1.400000	4	120.000000		3	0.000000
    C	6	1.340000	5	120.000000		4	180.000000
    C	1	1.340000	2	120.000000		3	180.000000
    C	8	1.340000	1	120.000000		2	180.000000
    C	7	1.340000	6	120.000000		5	180.000000
    O	7	1.240000	6	120.000000		5	0.000000
    H	11	0.890000	7	120.000000		6	180.000000
    O   5	1.240000	4	120.000000		3	180.000000
    H	13	0.890000	5	120.000000		6	180.000000
    O	3	1.240000	4	120.000000		5	180.000000
    H	15	0.890000	3	120.000000		4	180.000000
    O	8	1.240000	1	120.000000		6	180.000000
    O	9	1.240000	8	120.000000		1	180.000000
    H	17	0.890000	8	120.000000		9	180.000000
    H	18	0.890000	9	120.000000		8	180.000000
    H	10	0.990000	7	120.000000		6	180.000000
    F	2	1.390000	3	109.471000		4	180.000000
    F	4	1.390000	3	120.000000		2	180.000000"""

plot_molecule(zmat)
plt.savefig('Molecule.png', dpi=100)
plot_genome([180, 0, 180, 180, 0],zmat)
plt.savefig('Final_Molecule.png', dpi=100)
plt.show()

# initial values
possibleangles = [0, 60, 120, 180, 240, 300]  # [0, 180]
pop_size = 10
num_gen = 30
mut_prob = 0.8

# create directory to save figures and create output file
dirname = 'test_popsize_' + str(pop_size) + '_mut_prob_' + str(mut_prob) + '_3'
try:
    os.makedirs(dirname)
except OSError:
    pass  # already exists
sys.stdout = open(dirname + '/' + 'outfile_popsize_' + str(pop_size) + '_mutrate_' + str(mut_prob) + '.txt', "w")

# start genetic algorithm
start = time.time()
best_genome, best_energies, average_energies, explored_energies, init_pop, final_pop = ga(
    population_size=pop_size, num_generations=num_gen,
    genome_length=5, possible_angles=possibleangles,
    mutation_prob=mut_prob)
end = time.time()
runtime = end - start
print('runtime: %1.2f min' % (runtime/60))

# plot results
plot_genome(best_genome, zmat)
plt.savefig(dirname + '/' + 'BestGenome_popsize_' + str(pop_size) + '_mutrate_' + str(mut_prob) + '.png', dpi=100)

plotfunc(best_energies, average_energies, explored_energies, pop_size, mut_prob, dirname)
plt.show()

sys.stdout.close()
