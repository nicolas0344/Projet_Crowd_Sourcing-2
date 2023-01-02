#%%
from functions import *
  
#
  
user = "au"
  
# 
  
path_to_Github_folder, path_to_project, path_to_CIFAR10 = paths(user)
os.chdir(path_to_CIFAR10)
# print(os.listdir())
  
# Let's work on data_batch_1
data = unpickle(os.path.join(path_to_CIFAR10,"test_batch")) #images 
meta = unpickle(os.path.join(path_to_CIFAR10,"batches.meta")) #labels
  
labels = meta['label_names']  #nom des labels

X = data['data']   # images couleurs (10 000, 3072)
Y = data['labels']   # numéro de labélisation  10 000
  
n = 9911 #random.randrange(0,len(X))
display(X[n],Y[n],labels)

#%%
os.chdir(path_to_project)

df = pd.read_csv("cifar10h-raw.csv",na_values="-99999")
df.dropna(inplace=True)  # delete annotators/spamers

df_test = df[['annotator_id','true_label','chosen_label','cifar10_test_test_idx']]
df_test = df_test[df_test['cifar10_test_test_idx']<100]

print(len(df_test))
print(len(df_test['cifar10_test_test_idx'].unique()))
print(len(df_test['annotator_id'].unique()))
# 5136
# 100
# 2197
K = len(df_test['annotator_id'].unique())
I = len(df_test['cifar10_test_test_idx'].unique())
J = 10

idx = df['cifar10_test_test_idx'].unique()  #numéro de l'image
K = len(df['annotator_id'].unique())
# 2571 annotators

PI_k = []
N_k = []
for i in range(K):
    PI_k += [ np.ones((len(labels),len(labels))) ]
    N_k += [ np.ones((len(X),len(labels))) ]

# print(np.shape(PI_k[0]))
# (10, 10)

# PI_k[j][l] : in theory probability that the observer has given the label l
# where j is the right label

T = np.ones((len(X),len(labels)))
# print(np.shape(T))
# (10 000, 10)
# T[i][j] : in theory 1 if j is the correct label for i

p = [1]*len(labels)
# print(len(p))
# 10

# p[j] : in theory probability that a random image has the label j

#%%

PI_test = []
N_test = []
for k in range(1000):
    PI_test += [ np.random.random((5,5)) ]
    N_test += [ np.random.random((10000,5)) ]

T_test = np.random.random((10000,5))
p_test = [1]*5



K = 1000 #len(df_test['annotator_id'].unique())
I = 10000 #len(df_test['cifar10_test_test_idx'].unique())
J = 10

PI_test2 = []
N_test2 = []
for k in range(K):
    PI_test2 += [ np.random.random((J,J)) ]
    N_test2 += [ np.random.random((I,J)) ]

T_test2 = np.random.random((I,J))
p_test2 = [1]*J

df_test = df[['annotator_id','true_label','chosen_label','cifar10_test_test_idx']]
df_test = df_test[df_test['annotator_id']<60]
print(len(df_test['annotator_id'].unique()))
df_test = df_test[df_test['cifar10_test_test_idx']<100]
print(len(df_test['cifar10_test_test_idx'].unique()))
print(len(df_test['annotator_id'].unique()))


T_test = np.random.random((I,J))
N_test = []
for k in range(K):
    print(k)
    a = df_test[df_test['annotator_id'] == k]
    N_test += [ create_matrix_N(a) ]
    
def create_matrix_N(A):
    N = np.zeros((I,J))
    for i in list(A['cifar10_test_test_idx']):
        i = int(i)
        j = int(A[A['cifar10_test_test_idx'] == i]['chosen_label'])
        N[i,j] = 1
    return(N)



def produit_liste(A):
    a = 1
    for i in A: 
        a = a*i
    return(a)


def EM_stepM(T,N):
    I = len(T[:,0])
    J = len(T[0,:])
    K = len(N)
    
    PI = []
    for k in range(K):
        PI += [ np.ones((J,J)) ]
    p = [1]*J
    
    for k in range(K):
        for l in range(J): 
            
            sum_j = 0.0
            for a in range(J):
                sum_j += sum(T[:,l]*N[k][:,a]) #step E
            
            for j in range(J): 
                sum_i = sum(T[:,l]*N[k][:,j])
                PI[k][l,j] = sum_i/sum_j  #step E
            
            p[l] = sum(T[:,l])/I
    return([PI,p,N,T])

EM_stepM(T_test, N_test)


def EM_stepE(f):
    PI = f[0]
    p = f[1]
    N = f[2]
    T = f[3]
    
    I = len(N[0][:,0])
    J = len(N[0][0,:])
    K = len(N)
    
    for i in range(I):
        prod_2 = []
        for l in range(J):
            prod_1 = p[l]
            for k in range(K):
                for j in range(J):
                    prod_1 = prod_1 * ( PI[k][l,j] ) **(N[k][i,j])
            
            prod_2.append(prod_1)
        
        for l in range(J): 
            T[i,l] = prod_2[l]/sum(prod_2)
    return(T)
        
EM_stepE(EM_stepM(T_test,N_test))
EM_stepE(EM_stepM(T_test2,N_test2))
1 * 1 * 0.1**1.0

def EM_algo(T,N,n):
    
    for i in range(n):
        T = EM_stepE(EM_stepM(T,N))
    return(T)

EM_algo(T_test,N_test,1)

# %%
