import numpy as np

arr = np.array([1, 2, 3, 4, 5])  #1d array
arr2 = np.array([[1,2,3,4,5],[5,6,7,8,9]])  #2d array
arr3 = np.array([[[1,2,3,4],[4,5,6,7]],[[6,7,8,9],[9,8,7,6]]])  #3rd array

arrn = np.array([1,2,3,4,5],ndmin=5)  #nd array we are specifining the n-d array  with "ndmin=5"


print(arr)
print(arr2[0,1]) # accessing 0d , 2nd number
print(arr2.ndim)  # Tells us how many dynamisons has the array has
print(arr3[1,1,3]) #accessing 3rd dynamsion array
print(arr3[-2,1, 1]) # accesssing negative indexing of 3rd
print(arr3.ndim)
print(arrn)


k = np.zeros([2,3])
print(k)
print(type(arr2))
#1
print(np.__version__)

# 2. Creating Arrays
arr1 = np.array([1, 2, 3])        # array([1, 2, 3])
arr2 = np.zeros((2, 3))           # array([[0., 0., 0.],
                                 #        [0., 0., 0.]])
arr3 = np.ones((3, 3))            # array([[1., 1., 1.],
                                 #        [1., 1., 1.],
                                 #        [1., 1., 1.]])
arr4 = np.empty((2, 2))           # uninitialized values, e.g. array([[6.9e-310, 6.9e-310],
                                 #                                 [6.9e-310, 6.9e-310]])
arr5 = np.arange(0, 10, 2)        # array([0, 2, 4, 6, 8])
arr6 = np.linspace(0, 1, 5)       # array([0.  , 0.25, 0.5 , 0.75, 1.  ])
arr7 = np.full((2, 2), 7)         # array([[7, 7],
                                 #        [7, 7]])
arr8 = np.eye(3)                  # Identity matrix
                                 # array([[1., 0., 0.],
                                 #        [0., 1., 0.],
                                 #        [0., 0., 1.]])
arr9 = np.random.rand(2, 2)        # random floats in [0,1], e.g. array([[0.55, 0.12],
                                 #                                       [0.33, 0.44]])
arr10 = np.random.randint(1, 10, 5) # random ints, e.g. array([3, 7, 1, 8, 6])

# 3. Array Indexing
arr = np.array([10, 20, 30, 40])
print(arr[0])     # 10
print(arr[-1])    # 40
mat = np.array([[1, 2], [3, 4]])
print(mat[1, 0])  # 3

# 4. Array Slicing
arr = np.array([10, 20, 30, 40, 50])
print(arr[1:4])    # [20 30 40]
print(arr[::2])    # [10 30 50]
mat = np.array([[1, 2, 3], [4, 5, 6]])
print(mat[:, 1:])  # [[2 3]
                   #  [5 6]]

# 5. Data Types
arr = np.array([1, 2, 3])
print(arr.dtype)           # int64 (or int32 depending on system)
float_arr = arr.astype(float)
print(float_arr.dtype)     # float64

# 6. Copy vs View
arr = np.array([1, 2, 3])
copy_arr = arr.copy()
view_arr = arr.view()
arr[0] = 100
print("Copy:", copy_arr)   # Copy: [1 2 3]
print("View:", view_arr)   # View: [100   2   3]
print("Is View:", view_arr.base is arr)  # True

# 7. Array Shape
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape:", arr.shape)     # (2, 3)
print("Dimensions:", arr.ndim) # 2
print("Size:", arr.size)       # 6

# 8. Array Reshape
arr = np.arange(6)
reshaped = arr.reshape(2, 3)
print("Reshaped:\n", reshaped)  # [[0 1 2]
                                #  [3 4 5]]
print("Flatten:", reshaped.flatten())  # [0 1 2 3 4 5]

# 9. Array Iterating
arr = np.array([[1, 2], [3, 4]])
print("nditer:")
for x in np.nditer(arr):
    print(x)                  # 1 2 3 4 (each on new line)
print("ndenumerate:")
for idx, val in np.ndenumerate(arr):
    print(f"Index: {idx}, Value: {val}")
    # Index: (0, 0), Value: 1
    # Index: (0, 1), Value: 2
    # Index: (1, 0), Value: 3
    # Index: (1, 1), Value: 4

# 10. Array Join
a = np.array([1, 2])
b = np.array([3, 4])
print("Concatenate:", np.concatenate((a, b)))  # [1 2 3 4]
print("VStack:\n", np.vstack((a, b)))          # [[1 2]
                                                #  [3 4]]
print("HStack:\n", np.hstack((a, b)))          # [1 2 3 4]

# 11. Array Split
arr = np.array([1, 2, 3, 4, 5, 6])
print("Split:", np.split(arr, 3))               # [array([1, 2]), array([3, 4]), array([5, 6])]
mat = np.array([[1, 2], [3, 4], [5, 6]])
print("VSplit:", np.vsplit(mat, 3))
# [array([[1, 2]]), array([[3, 4]]), array([[5, 6]])]

# 12. Array Search
arr = np.array([1, 2, 3, 4, 5])
print("Where > 2:", np.where(arr > 2))         # (array([2, 3, 4]),)
print("Searchsorted(3):", np.searchsorted(arr, 3))  # 2
print("Non-zero:", np.nonzero(arr))             # (array([0, 1, 2, 3, 4]),)

# 13. Array Sort
arr = np.array([5, 1, 4, 2])
print("Sorted:", np.sort(arr))                   # [1 2 4 5]
print("Argsort:", np.argsort(arr))               # [1 3 2 0]

# 14. Array Filter
arr = np.array([10, 15, 20, 25])
filter = arr > 18
print("Filtered:", arr[filter])                   # [20 25]

# 15. Random Module
print("Random rand(2,2):\n", np.random.rand(2, 2))
# e.g. [[0.5488135  0.71518937]
#       [0.60276338 0.54488318]]
print("Random permutation(10):", np.random.permutation(10))
# e.g. [2 8 4 1 9 6 3 0 7 5]
print("Random integers:", np.random.randint(1, 100, 5))
# e.g. [55 12 65 89 70]

# 16. Universal Functions (ufuncs)
arr = np.array([1, 2, 3, 4])
print("sqrt:", np.sqrt(arr))                      # [1.         1.41421356 1.73205081 2.        ]
print("sin:", np.sin(arr))                        # [0.84147098 0.90929743 0.14112001 -0.7568025 ]
print("exp:", np.exp(arr))                        # [ 2.71828183  7.3890561  20.08553692 54.59815003]

# 17. Linear Algebra
matrix = np.array([[1, 2], [3, 4]])
vector = np.array([5, 6])
print("Inverse:\n", np.linalg.inv(matrix))
# [[-2.   1. ]
#  [ 1.5 -0.5]]
print("Solve:", np.linalg.solve(matrix, vector))  # [ -4.   4.5]

# 18. Statistical Functions
data = np.array([1, 2, 3, 4, 5])
print("Mean:", np.mean(data))                      # 3.0
print("Median:", np.median(data))                  # 3.0
print("Standard Deviation:", np.std(data))        # 1.4142135623730951

# 19. Set Operations
a = np.array([1, 2, 3, 4])
b = np.array([3, 4, 5, 6])
print("Union:", np.union1d(a, b))                  # [1 2 3 4 5 6]
print("Intersection:", np.intersect1d(a, b))       # [3 4]
print("Difference:", np.setdiff1d(a, b))           # [1 2]
