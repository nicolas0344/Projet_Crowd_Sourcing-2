from functions import *

#

user = "au"

# 

path_to_Github_folder, path_to_project, path_to_data = paths(user)

os.chdir(path_to_data)
# print(os.listdir())

# Let's work on the test_batch

file = os.path.join(path_to_data,"test_batch")
data = unpickle(file)
meta_file = os.path.join(path_to_data,"batches.meta")
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

show = True

if show == True :

    X = data['data']
    Y = data['labels']

    display(X[0],Y[0],labels)

