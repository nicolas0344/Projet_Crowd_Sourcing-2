#%%
from functions import *
from source import *
  

# inter your name user difined in the paths function of the function file
user = "nicolas"
  
#%%
  
# display an image from data set

path_to_Github_folder, path_to_project, path_to_CIFAR10 = paths(user)
os.chdir(path_to_CIFAR10)
# print(os.listdir())
  
# Let's work on data_batch
data = unpickle(os.path.join(path_to_CIFAR10,"test_batch")) #images 
meta = unpickle(os.path.join(path_to_CIFAR10,"batches.meta")) #labels
  
labels = meta['label_names']  #name of labels

X = data['data']   # images colors (10 000, 3072)
Y = data['labels']   # numerotation of labelisation  10 000
  
n = 9911 #random.randrange(0,len(X))
display(X[n],Y[n],labels) #image 

#%%

# Define data set of paticipants and their responses

os.chdir(path_to_project)

df = pd.read_csv("cifar10h-raw.csv",na_values="-99999")
df.dropna(inplace=True)  # delete annotators/spamers
data = df[['annotator_id','true_label','chosen_label','cifar10_test_test_idx']]


# Data set size is reduced for digital purposes

data_test = data[data['annotator_id']<150]
print(len(data_test['annotator_id'].unique()))
print(len(data_test['cifar10_test_test_idx'].unique()))
data_test = data_test[data_test['cifar10_test_test_idx']<250]
print(len(data_test['cifar10_test_test_idx'].unique()))
print(len(data_test['annotator_id'].unique()))

#%%

# Initialisation of parameters

K = len(data_test['annotator_id'].unique())
I = len(data_test['cifar10_test_test_idx'].unique())
J = 10

T_test = np.ones((I,J))*(1/J)
N_test = []
for k in range(K):
    print(k)
    a = data_test[data_test['annotator_id'] == k]
    N_test += [ create_matrix_N(a,I,J) ]



# PI[k][j,l] : in theory probability that the observer has given the label l
# where j is the right label

# T[i][j] : in theory 1 if j is the correct label for i

# p[j] : in theory probability that a random image has the label j

#%%

# if we want to try others values

# K = ..
# I = ..
# J = ..

# N_test = []
# for k in range(K):
#     N_test += [ .. numpy matrix of size = (I,J) .. ]

# T_test = .. numpy matrix of size = (I,J) ..
# p_test =  .. list of size = J .. 


#%%

# Execution of EM_algorithm

EM_stepM(T_test, N_test)
        
EM_stepE(EM_stepM(T_test,N_test))

n = 3 # interation 
EM_algo(T_test,N_test,n)

# here we note that the matrix of the T's is the same that the initialization
# EM_stepE(EM_stepM(T_test,N_test))
# it's a problem, and haven't fixed this 


