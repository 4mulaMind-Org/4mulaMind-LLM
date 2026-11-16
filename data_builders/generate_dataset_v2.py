from pathlib import Path


# ============================================================
# 4mulaMind Dataset V2 Generator
# ============================================================

OUTPUT_PATH = Path("data/train.txt")

TARGET_WORDS = 80000


# ============================================================
# Dataset Content
# ============================================================

sections = [

# ------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------

"""
A tokenizer converts raw text into a sequence of tokens.
Each token is represented by an integer identifier.
The model receives token identifiers as input.
Token embeddings convert token identifiers into vectors.
Position embeddings provide information about token order.
Encoding converts text into token identifiers.
Decoding converts token identifiers back into text.
A vocabulary contains the tokens that a model can represent.
A tokenizer should produce deterministic token sequences.
A trained tokenizer should be saved with its vocabulary.
Loading a tokenizer should preserve its encoding behavior.
Unknown tokens represent symbols that are not present in the vocabulary.
A tokenizer is part of the input pipeline of a language model.
Subword tokenization can represent common words efficiently.
Rare words can be represented using multiple subword tokens.
Tokenization affects the sequence length seen by the model.
A smaller vocabulary can require more tokens for the same sentence.
A larger vocabulary can represent longer text fragments.
""",

# ------------------------------------------------------------
# BPE
# ------------------------------------------------------------

"""
Byte pair encoding is a subword tokenization algorithm.
BPE begins with a vocabulary containing basic symbols.
Adjacent token pairs are counted during BPE training.
The most frequent pair can be merged into a new token.
The merge operation reduces repeated patterns.
BPE merges are learned from a training corpus.
The order of BPE merges is important.
The tokenizer must apply learned merges in the same order.
Common words can become single tokens after sufficient training.
Rare words can be represented by several subword tokens.
BPE can reduce sequence length compared with character tokenization.
The vocabulary and merge rules must be saved together.
A tokenizer test should verify encoding and decoding.
BPE training should use representative text from the dataset.
Changing the tokenizer vocabulary can require model retraining.
The model vocabulary size must match the tokenizer vocabulary size.
""",

# ------------------------------------------------------------
# Neural Networks
# ------------------------------------------------------------

"""
A neural network learns parameters from training data.
Parameters are adjusted during optimization.
Gradients describe how the loss changes with respect to parameters.
Backpropagation computes gradients through the neural network.
An optimizer uses gradients to update model parameters.
The learning rate controls the size of parameter updates.
A large learning rate can make training unstable.
A small learning rate can make training slower.
The forward pass produces model predictions.
The loss function compares predictions with expected targets.
The backward pass computes gradients.
The optimizer updates parameters using those gradients.
Training is repeated for many optimization steps.
A batch contains multiple training examples.
Batch size determines how many examples are processed together.
An epoch represents a complete pass through a dataset.
A checkpoint stores the state of a trained model.
Validation loss can be used to select the best checkpoint.
Early stopping can prevent unnecessary training.
""",

# ------------------------------------------------------------
# Transformer
# ------------------------------------------------------------

"""
A transformer is a neural network architecture based on attention.
Transformers process sequences using attention mechanisms.
Self attention calculates relationships between tokens.
Queries, keys, and values are used to calculate attention.
A query represents information that a token is looking for.
A key represents information that can be matched against a query.
A value contains information that can be transferred.
Attention scores are calculated between queries and keys.
The scores are normalized to produce attention weights.
Weighted values are combined to produce attention outputs.
Multi head attention uses several attention mechanisms in parallel.
Different attention heads can learn different relationships.
The outputs of attention heads are combined.
A projection layer transforms the combined representation.
Feed forward layers process token representations.
Residual connections help information flow through transformer layers.
Layer normalization helps stabilize transformer training.
Position information is required because attention alone does not encode order.
Causal attention prevents a language model from seeing future tokens.
A causal mask blocks attention to later tokens.
Causal language models generate text from left to right.
Context length determines how many previous tokens the model can use.
""",

# ------------------------------------------------------------
# Language Modeling
# ------------------------------------------------------------

"""
A language model predicts tokens based on context.
The context contains tokens that appear before the predicted token.
The model learns statistical relationships between tokens.
Training adjusts parameters to reduce prediction loss.
A transformer can model long range relationships using attention.
A prompt provides the initial context for generation.
The model generates one token at a time.
The generated token becomes part of the next context.
Generation continues until a stopping condition is reached.
Maximum generation length limits generated tokens.
Cross entropy loss measures the difference between predicted and target tokens.
A lower validation loss generally indicates better generalization.
Training loss measures performance on training examples.
Validation loss measures performance on held out examples.
A clean dataset can improve language model quality.
Repeated examples can cause memorization.
Generated text should contain meaningful words and sentence structure.
""",

# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

"""
Logits are unnormalized scores produced by the model.
Softmax converts logits into probabilities.
The highest probability token is one possible next token.
Greedy decoding selects the most probable token.
Sampling can generate different outputs from the same prompt.
Temperature controls the sharpness of the probability distribution.
A lower temperature generally produces conservative outputs.
A higher temperature generally produces more varied outputs.
Top k sampling restricts choices to the k most likely tokens.
Top p sampling restricts choices to a probability mass.
Repetition penalties can reduce repeated tokens.
Generation should prevent invalid token identifiers.
The generator should tokenize the prompt before generation.
The generator should decode generated identifiers back into text.
A good generation system should preserve the original prompt.
""",

# ------------------------------------------------------------
# Python
# ------------------------------------------------------------

"""
Python is a general purpose programming language.
Python is widely used for machine learning and numerical computing.
Python provides simple syntax and dynamic typing.
A variable stores a reference to a value.
A list stores an ordered collection of values.
A tuple stores an immutable collection of values.
A dictionary stores key value pairs.
A set stores unique values.
A function groups reusable operations.
A class defines the structure and behavior of objects.
An object is an instance of a class.
Inheritance allows a class to reuse behavior.
Encapsulation combines data and related operations.
Polymorphism allows different objects to share an interface.
Exception handling allows programs to handle runtime errors.
A loop repeats a block of instructions.
A conditional statement chooses an execution path.
A module contains reusable Python code.
A package organizes related Python modules.
A virtual environment isolates project dependencies.
A package manager installs external libraries.
""",

# ------------------------------------------------------------
# PyTorch
# ------------------------------------------------------------

"""
PyTorch is a machine learning framework.
PyTorch provides tensors and automatic differentiation.
A tensor can contain scalar, vector, matrix, or higher dimensional data.
Tensor operations can be performed efficiently on supported hardware.
Autograd computes gradients automatically.
A model can inherit from the PyTorch module class.
Model parameters can be optimized using an optimizer.
The Adam optimizer is commonly used for neural networks.
AdamW is a variant that decouples weight decay.
PyTorch provides linear layers and embedding layers.
A tensor has a shape that describes its dimensions.
Tensor shapes must be compatible for many operations.
Reshaping changes the organization of tensor elements.
Matrix multiplication combines compatible dimensions.
The device determines where a tensor is stored.
A model can run on a CPU or accelerator.
Training mode enables training specific behavior.
Evaluation mode disables training specific behavior.
Saving a model allows it to be loaded later.
A state dictionary contains model parameter values.
""",

# ------------------------------------------------------------
# C++
# ------------------------------------------------------------

"""
C++ is a compiled general purpose programming language.
C++ supports procedural programming and object oriented programming.
A variable has a type and a value.
An integer stores whole numbers.
A floating point value stores fractional numbers.
A pointer stores a memory address.
A reference provides another name for an existing object.
A class can contain data members and member functions.
A constructor initializes an object.
A destructor is called when an object is destroyed.
Inheritance allows one class to derive from another.
Virtual functions support runtime polymorphism.
Templates allow generic programming.
The standard template library provides data structures and algorithms.
A vector is a dynamic array.
A stack follows the last in first out principle.
A queue follows the first in first out principle.
A priority queue returns elements according to priority.
A map stores key value associations.
An unordered map uses hashing.
A set stores unique ordered values.
An iterator traverses elements in a container.
RAII connects resource lifetime with object lifetime.
""",

# ------------------------------------------------------------
# DSA
# ------------------------------------------------------------

"""
A data structure organizes data for efficient access and modification.
An array stores elements in contiguous memory.
An array provides constant time access by index.
A linked list consists of nodes connected by links.
A singly linked list stores a link to the next node.
A doubly linked list stores links in both directions.
A stack supports push and pop operations.
A queue supports insertion and removal operations.
A deque supports insertion and removal at both ends.
A tree is a hierarchical data structure.
A binary tree has at most two children per node.
A binary search tree maintains an ordering property.
A balanced tree keeps its height relatively small.
A heap is commonly used to implement a priority queue.
A graph consists of vertices and edges.
A directed graph contains directed edges.
An undirected graph contains edges without direction.
A weighted graph associates values with edges.
An adjacency list stores neighboring vertices.
An adjacency matrix stores graph connections.
Breadth first search explores a graph level by level.
Depth first search explores a path before backtracking.
""",

# ------------------------------------------------------------
# Searching and Sorting
# ------------------------------------------------------------

"""
Binary search repeatedly divides a sorted search space.
The middle element is compared with the target value.
If the target is smaller, search continues in the left half.
If the target is larger, search continues in the right half.
Binary search has logarithmic time complexity.
Binary search requires an ordered search space.
Linear search checks elements one by one.
Linear search has linear time complexity.
Sorting arranges elements according to an ordering rule.
Merge sort divides an array into smaller parts.
Merge sort combines sorted parts into a larger sorted sequence.
Merge sort has O of n log n time complexity.
Quick sort selects a pivot and partitions the array.
Quick sort performance depends on pivot selection.
Insertion sort builds a sorted portion one element at a time.
Selection sort repeatedly selects the smallest remaining element.
Bubble sort exchanges adjacent elements when they are out of order.
Hashing maps keys to positions using a hash function.
Hash collisions occur when multiple keys map to one position.
Big O notation describes how an algorithm scales with input size.
Constant time is O of one.
Linear time is O of n.
Logarithmic time is O of log n.
Quadratic time is O of n squared.
""",

# ------------------------------------------------------------
# Database
# ------------------------------------------------------------

"""
A database stores persistent information.
A relational database organizes information into tables.
A table contains rows and columns.
A row represents one record.
A column represents an attribute.
A primary key uniquely identifies a row.
A foreign key connects related tables.
A relational schema describes database structure.
SQL is used to query relational databases.
A select statement retrieves data.
A where clause filters records.
An order by clause sorts query results.
A group by clause groups records.
An aggregate function computes values from multiple rows.
Count returns the number of rows.
A join combines records from related tables.
An inner join returns matching records.
A left join preserves records from the left table.
An index can improve query performance.
Transactions group related database operations.
Atomicity means a transaction is completed fully or not at all.
Isolation controls concurrent transaction interaction.
Durability means committed changes remain persistent.
Normalization reduces unnecessary data duplication.
""",

# ------------------------------------------------------------
# Operating Systems
# ------------------------------------------------------------

"""
An operating system manages computer hardware and software resources.
A process is a program in execution.
A thread is a unit of execution within a process.
Processes can have separate memory spaces.
Threads within a process can share memory.
The operating system schedules processes and threads.
CPU scheduling determines which task runs next.
Context switching changes the active execution context.
Virtual memory provides an abstraction over physical memory.
Pages divide virtual memory into fixed size blocks.
A page fault occurs when a required page is not in memory.
Memory management tracks allocation and release of memory.
A file system organizes files and directories.
A system call provides an interface between applications and the operating system.
The operating system manages input and output devices.
Deadlock occurs when processes wait indefinitely for resources.
Mutual exclusion is one condition associated with deadlock.
Synchronization coordinates access to shared resources.
A mutex can protect a critical section.
A critical section accesses shared state requiring synchronization.
""",

# ------------------------------------------------------------
# Computer Networks
# ------------------------------------------------------------

"""
A computer network connects devices so they can exchange information.
A network protocol defines rules for communication.
The Internet is a global network of interconnected systems.
An IP address identifies a network interface.
IPv4 uses 32 bit addresses.
IPv6 uses 128 bit addresses.
A router forwards packets between networks.
A switch connects devices within a local network.
TCP provides reliable ordered byte stream communication.
UDP provides connectionless datagram communication.
HTTP is an application layer protocol.
HTTPS protects HTTP traffic using encryption.
DNS maps domain names to IP addresses.
A port identifies a communication endpoint.
A packet contains data and control information.
Network latency represents communication delay.
Bandwidth represents the amount of data transferred over time.
Client server architecture separates clients from service providers.
A server listens for incoming requests.
A client sends requests to a server.
Sockets provide an interface for network communication.
""",

# ------------------------------------------------------------
# Software Engineering
# ------------------------------------------------------------

"""
Software engineering applies systematic methods to software development.
A software project begins with requirements.
Functional requirements describe what a system should do.
Non functional requirements describe quality attributes and constraints.
Software architecture defines major system components.
A module should have a clear responsibility.
High cohesion keeps related responsibilities together.
Low coupling reduces unnecessary dependencies.
An interface defines how components communicate.
Object oriented design models software using objects and classes.
A design pattern provides a reusable solution to a common problem.
The singleton pattern restricts a class to one instance.
The factory pattern separates object creation from object usage.
The observer pattern allows objects to react to changes.
The strategy pattern allows algorithms to be exchanged.
Unit testing verifies small pieces of software.
Integration testing verifies component interactions.
System testing evaluates the complete system.
Debugging identifies and fixes software defects.
Version control tracks source code changes.
Git is a distributed version control system.
A commit records a set of changes.
A branch provides an independent development line.
Code review helps improve software quality.
""",

# ------------------------------------------------------------
# Machine Learning
# ------------------------------------------------------------

"""
Machine learning enables systems to learn patterns from data.
Supervised learning uses labeled examples.
Unsupervised learning finds patterns without target labels.
Classification predicts discrete categories.
Regression predicts continuous values.
Training data is used to learn model parameters.
Features represent measurable properties of examples.
Labels represent target outputs.
A model maps inputs to predictions.
Overfitting occurs when a model learns training data too closely.
Underfitting occurs when a model is too simple.
Regularization can reduce overfitting.
A training set is used for learning.
A validation set is used for model selection.
A test set is used for final evaluation.
Accuracy measures the fraction of correct predictions.
Precision measures how many predicted positives are correct.
Recall measures how many actual positives are detected.
The F1 score combines precision and recall.
A confusion matrix summarizes classification predictions.
Gradient descent minimizes an objective function.
Learning rate is an important optimization hyperparameter.
Feature scaling can improve some machine learning algorithms.
""",

# ------------------------------------------------------------
# Mathematics
# ------------------------------------------------------------

"""
Probability measures uncertainty.
A probability value lies between zero and one.
An event is a possible outcome or collection of outcomes.
A random variable assigns numerical values to outcomes.
Expected value represents the average outcome.
Variance measures the spread of a distribution.
Standard deviation is the square root of variance.
A probability distribution describes probability assigned to values.
Conditional probability measures probability given information.
Bayes theorem relates conditional probabilities.
Statistics provides methods for understanding data.
The mean is the arithmetic average.
The median is the middle value after sorting.
The mode is the most frequent value.
A sample is a subset of a population.
Sampling is used when collecting the entire population is expensive.
Correlation measures the strength of a linear relationship.
Correlation does not necessarily imply causation.
A matrix is a rectangular arrangement of numbers.
A vector can represent a point or direction.
Matrix multiplication is important in machine learning.
A derivative measures the rate of change of a function.
A gradient contains partial derivatives.
Optimization searches for parameters that minimize a loss function.
""",

# ------------------------------------------------------------
# Algorithms
# ------------------------------------------------------------

"""
An algorithm is a finite sequence of steps for solving a problem.
A good algorithm should be correct and efficient.
Correctness means the algorithm produces the expected result.
Efficiency considers time and memory requirements.
Problem decomposition divides a complex problem into smaller tasks.
Abstraction hides unnecessary implementation details.
Modularity divides a system into independent components.
Recursion occurs when a function calls itself.
A recursive algorithm requires a base case.
Dynamic programming stores results of overlapping subproblems.
Memoization caches results to avoid repeated computation.
Greedy algorithms make locally optimal choices.
Graph algorithms solve many connectivity problems.
Shortest path algorithms find minimum cost routes.
Dijkstra algorithm works with non negative edge weights.
A minimum spanning tree connects vertices with minimum total weight.
Topological sorting orders vertices of a directed acyclic graph.
An acyclic graph contains no cycle.
A cycle returns to an earlier vertex.
Complexity analysis helps compare alternative algorithms.
""",

# ------------------------------------------------------------
# 4mulaMind
# ------------------------------------------------------------

"""
4mulaMind is an experimental language model project.
4mulaMind focuses on understanding how language models work internally.
The project includes tokenization, datasets, transformers, training, and generation.
The tokenizer converts text into numerical identifiers.
The dataset creates training examples from token sequences.
The model predicts the next token.
The loss function measures prediction error.
The trainer controls the optimization process.
The generator converts model predictions into generated text.
A checkpoint stores learned model parameters.
Validation loss helps select the best checkpoint.
Experiments should record model configuration and evaluation results.
A reproducible training pipeline makes debugging easier.
Software components should have clear responsibilities.
The tokenizer should be independent from the model architecture.
The dataset should provide input and target sequences.
The trainer should perform training and validation.
The generator should handle tokenization and decoding.
A language model is built from many interacting components.
Understanding each component helps explain the complete system.
A small language model is useful for learning machine learning engineering.
Model quality should be evaluated using numerical metrics and generated examples.
The goal of an educational language model is to make learning understandable.
A strong foundation in algorithms helps implement machine learning systems.
A strong foundation in mathematics helps explain optimization.
A strong foundation in programming helps build reliable machine learning software.
Experimentation is an important part of machine learning development.
Good engineering requires testing before changing multiple components.
"""
]


# ============================================================
# Build Base Dataset
# ============================================================

base_text = "\n".join(
    section.strip()
    for section in sections
)


# ============================================================
# Expand Dataset
#
# We generate controlled variations instead of simply
# duplicating the exact same text.
# ============================================================

templates = [

    "The concept of {sentence}",
    "In machine learning, {sentence}",
    "In a language model, {sentence}",
    "During training, {sentence}",
    "During inference, {sentence}",
    "In software engineering, {sentence}",
    "A practical implementation should remember that {sentence}",
    "When building a reliable system, {sentence}",
    "For an efficient implementation, {sentence}",
    "A developer should understand that {sentence}",
]


sentences = [
    line.strip()
    for line in base_text.splitlines()
    if line.strip()
]


dataset = []


# Add original sentences first
dataset.extend(sentences)


# Generate variations
round_number = 0

while len(" ".join(dataset).split()) < TARGET_WORDS:

    for sentence in sentences:

        template = templates[
            round_number % len(templates)
        ]

        dataset.append(
            template.format(
                sentence=sentence.lower()
            )
        )

        round_number += 1

        current_words = len(
            " ".join(dataset).split()
        )

        if current_words >= TARGET_WORDS:
            break


# ============================================================
# Write Dataset
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    for sentence in dataset:

        file.write(
            sentence.strip()
            + "\n"
        )


# ============================================================
# Statistics
# ============================================================

final_text = OUTPUT_PATH.read_text(
    encoding="utf-8"
)

character_count = len(final_text)

word_count = len(
    final_text.split()
)


print()
print("================================")
print("4mulaMind Dataset V2")
print("================================")
print()

print("Dataset:")
print(OUTPUT_PATH)

print()

print("Characters:")
print(character_count)

print()

print("Words:")
print(word_count)

print()

print("Approximate tokens:")
print(
    int(word_count * 1.3)
)

print()

print("Sentences:")
print(len(dataset))

print()

print("✓ Dataset V2 generated successfully")
print()
