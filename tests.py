from functions import *

#

user = "au"

# 

path_to_Github_folder, path_to_project, path_to_CIFAR10 = paths(user)
os.chdir(path_to_CIFAR10)
# print(os.listdir())

# Let's work on the test_batch

file = os.path.join(path_to_CIFAR10,"test_batch")
data = unpickle(file)
meta_file = os.path.join(path_to_CIFAR10,"batches.meta")
meta = unpickle(meta_file)

labels = meta['label_names']
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Type of data :

show = False

if show == True :

    print(type(data))
    print(data.keys())
    # <class 'dict'>
    # dict_keys(['batch_label', 'labels', 'data', 'filenames'])

    for i in data:
        print(i, type(data[i]))

    # batch_label <class 'str'>
    # labels <class 'list'>
    # data <class 'numpy.ndarray'>
    # filenames <class 'list'>

    print("Labels:", set(data['labels']))
    print("Label Names:", meta['label_names'] )

    # Labels: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    # Label Names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    X = data['data']

    print(X)
    # [[158 159 165 ... 124 129 110]
    # [235 231 232 ... 178 191 199]
    # [158 158 139 ...   8   3   7]
    # ...
    # [ 20  19  15 ...  50  53  47]
    # [ 25  15  23 ...  80  81  80]
    # [ 73  98  99 ...  94  58  26]]
    print(np.shape(X))
    # (10 000, 3072)
    print(len(X[1]))
    # 3072
    # each of the 10 000 images has 3,072 entries = 1024*3 (RGB)
    # first 1024 for Red, then 1024 for Green and last 1024 for Blue

    image = X[0]
    image = image.reshape(3,32,32)
    print(np.shape(image))
    # (3, 32, 32)
    # i.e. 3 matrixes 32x32
    image = image.transpose(1,2,0)
    print(image.shape)
    # (32, 32, 3)
    # to use matplotlib

# To display images :

show = False

if show == True :

    X = data['data']
    Y = data['labels']

    display(X[0],Y[0],labels)

# Let's look at CIFAR10h :

show = 1

if show == True :

    os.chdir(path_to_project)
    counts = np.load('cifar10h-counts.npy')
    df_counts = pd.DataFrame(counts)


    # print(type(counts))
    # <class 'numpy.ndarray'>

    # print(np.shape(counts))
    # (10 000, 10)

    # print(counts[1:100,])

    probs = np.load("cifar10h-probs.npy")
    df_probs = pd.DataFrame(probs)

    # print(type(probs))
    # <class 'numpy.ndarray'>

    # print(np.shape(probs))
    # (10 000, 10)

    print(df_counts[1:10])
    print(df_probs[1:10])

    df = pd.read_csv("cifar10h-raw.csv")

    # print(len(df))
    # 539 910

    # print(df.columns)

    print(df["annotator_id"].unique())

    print(
        len(
            df["annotator_id"].unique()
            )
        )

    # 2571 annotators

# Test of the annotator_matrix function :

show = False

if show == True :
    
    annotator_id = 0

    result = annotator_matrix(annotator_id,path_to_project)

    print(result)

# Tests divers :

show = 0

if show == True :

    os.chdir(path_to_project)
    df = pd.read_csv("cifar10h-raw.csv")

    #print(len(df))

    print(
        len(
            df["image_filename"].unique()
            )
        )

    probs = np.load("cifar10h-probs.npy")
    df_probs = pd.DataFrame(probs)

    print(len(df_probs))

    # The order of the 10,000 labels matches the original CIFAR-10 test set order.
    # ???

    # print(df.loc[df['annotator_id'] == 0,"image_filename"])
    # il y en a seulement 200 !

    # print(
    #     len(
    #         df.loc[df['annotator_id'] == 0,"image_filename"].unique()
    #         )
    #     )