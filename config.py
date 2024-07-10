
## DATASETS ##
TRAIN_DATASET = 'bottle' # Dataset to use for training
TEST_DATASET = 'bottle' # Dataset to use for predictions (testing)

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

## MISC ##
STN_CHECK = False # Run the STN sanity check step
MODEL_DIR = 'logs/models/' # Directory to save models
PLOTS_DIR = 'logs/plots' # Directory to save plots
GEN_DIR = 'logs/generated/' # Directory to save predictions
TEST_EPOCH = 1 # Test saved model at X epoch