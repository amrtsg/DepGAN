
## DATASETS ##
TRAIN_DATASET = 'bottle_basket' # Dataset to use for training
TEST_DATASET = 'bottle_basket' # Dataset to use for predictions (testing)
# The dataset is the dataset folder name inside 'data'

## GUI ##
USE_GRAPHS = True

## HYPERPARAMS ##
USE_VAL = True # Allow for validation steps during training
VAL_SAMPLES = 3 # Number of validation samples
VAL_FREQUENCY = 100 # Perform validation every x step
EPOCHS = 1000 # Number of epochs
BATCH_SIZE = 1 # Batch size
D_LR = 0.0001 # Discriminator learning rate
G_LR = 0.0002 # Generator learning rate

## ARCH PARAMS ##
USE_STN = False # Use STN in training
USE_DAL = True # Use Depth Aware Loss in the generator
EXTRA_DISC_LAYER = False # Use extra discriminator layer

## MISC ##
STN_CHECK = False # Run the STN sanity check step
MODEL_DIR = 'logs/models' # Directory to save models
PLOTS_DIR = 'logs/plots' # Directory to save plots
GEN_DIR = 'logs/generated' # Directory to save predictions
TEST_EPOCH = 5 # Test saved model at X epoch