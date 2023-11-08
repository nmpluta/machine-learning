# Cats vs Dogs labeled dataset is located in the ./dogs-vs-cats/train folder
# This script will split the dataset into training, validation and test sets.
# Training set with 1000 samples of each class, a validation set with 500 samples
# of each class, and finally a test set with 500 samples of each class.abs

import os, shutil
training_num = 1000
validation_num = 500
test_num = 500

# cat class directory name
cat_dir = 'class_a'

# dog class directory name
dog_dir = 'class_b'

# Create new directories for the training, validation and test sets
base_dir = './cats_and_dogs_small'
# Check if exists if yes delete it
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Sets for class A
train_a_dir = os.path.join(train_dir, cat_dir)
os.mkdir(train_a_dir)
validation_a_dir = os.path.join(validation_dir, cat_dir)
os.mkdir(validation_a_dir)
test_a_dir = os.path.join(test_dir, cat_dir)
os.mkdir(test_a_dir)

# Sets for class B
train_b_dir = os.path.join(train_dir, dog_dir)
os.mkdir(train_b_dir)
validation_b_dir = os.path.join(validation_dir, dog_dir)
os.mkdir(validation_b_dir)
test_b_dir = os.path.join(test_dir, dog_dir)
os.mkdir(test_b_dir)

training_num_a = training_num
validation_num_a = validation_num
test_num_a = test_num
training_num_b = training_num
validation_num_b = validation_num
test_num_b = test_num

# Randomly distribute the images into the training, validation and test sets
# Same images should not be in more than one set.
labeled_dataset_dir = './dogs-vs-cats/train'
for files in os.listdir(labeled_dataset_dir):
    if files.startswith('cat'):
        cat_file = os.path.join(labeled_dataset_dir, files)
        if training_num_a > 0:
            shutil.copyfile(cat_file, os.path.join(train_a_dir, files))
            training_num_a -= 1
        elif validation_num_a > 0:
            shutil.copyfile(cat_file, os.path.join(validation_a_dir, files))
            validation_num_a -= 1
        elif test_num_a > 0:
            shutil.copyfile(cat_file, os.path.join(test_a_dir, files))
            test_num_a -= 1
    elif files.startswith('dog'):
        dog_file = os.path.join(labeled_dataset_dir, files)
        if training_num_b > 0:
            shutil.copyfile(dog_file, os.path.join(train_b_dir, files))
            training_num_b -= 1
        elif validation_num_b > 0:
            shutil.copyfile(dog_file, os.path.join(validation_b_dir, files))
            validation_num_b -= 1
        elif test_num_b > 0:
            shutil.copyfile(dog_file, os.path.join(test_b_dir, files))
            test_num_b -= 1
    else:
        print('File name does not start with cat or dog: ' + files)
        exit(1)

# Print the number of images in each set for each class
print('Total number of images in training set for class A: ' + str(training_num))
print('Total number of images in validation set for class A: ' + str(validation_num))
print('Total number of images in test set for class A: ' + str(test_num))
print('Total number of images in training set for class B: ' + str(training_num))
print('Total number of images in validation set for class B: ' + str(validation_num))
print('Total number of images in test set for class B: ' + str(test_num))
