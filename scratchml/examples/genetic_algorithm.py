
from sratchml.unsupervised_learning import GeneticAlgorithm

def main():
    target_string = "Genetic Algorithm"
    population_size = 100
    mutation_rate = 0.05
    genetic_algorithm = GeneticAlgorithm(target_string,
                                        population_size,
                                        mutation_rate)
    print ("Parameters")
    print ("----------")
    print ("Target String: '%s'" % target_string)
    print ("Population Size: %d" % population_size)
    print ("Mutation Rate: %s" % mutation_rate)
    print ("")

    genetic_algorithm.run(iterations=1000)

if __name__ == "__main__":
    main()
