from detectron2.config import CfgNode

def add_custom_params(cfg):
    """
    In order to add custom config parameter in the .yaml those parameter must
    be initialised
    """
    # Model
    cfg.MODEL_CUSTOM = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE = CfgNode()
    cfg.MAX_EPOCHS= 700
    #Dataset
    cfg.ANOMALY_DATASET = CfgNode()
    cfg.ANOMALY_DATASET.MAX_SAMPLES = 100
    cfg.ANOMALY_DATASET.PREVIEW_DATASET = True
    cfg.ANOMALY_DATASET.SHUFFLE = True
    cfg.ANOMALY_DATASET.MAX_SIZE = 1500
    cfg.ANOMALY_DATASET.ANOMALY_FOLDER = ""  
    cfg.ANOMALY_DATASET.DATASET_PATH = CfgNode()
    cfg.ANOMALY_DATASET.DATASET_PATH.ROOT = "datasets"
    cfg.ANOMALY_DATASET.DATASET_PATH.ANOMALY_SOURCE = ""
    cfg.ANOMALY_DATASET.DATASET_PATH.MVTEC = ""
    

    cfg.TREES_DATASET = CfgNode()
    cfg.TREES_DATASET.MAX_SAMPLES = 100
    cfg.TREES_DATASET.PREVIEW_DATASET = True
    cfg.TREES_DATASET.SHUFFLE = True
    cfg.TREES_DATASET.MAX_SIZE = 1500
    cfg.TREES_DATASET.DATASET_PATH = CfgNode()

    cfg.TREES_DATASET.DATASET_PATH.ROOT = "datasets/synth_anomaly_May2023"
    cfg.TREES_DATASET.DATASET_PATH.RGB_TRAIN = "train/augmented"
    cfg.TREES_DATASET.DATASET_PATH.RGB_VALID = "val/augmented"
    cfg.TREES_DATASET.DATASET_PATH.RGB_TEST = "test/augmented"
    cfg.TREES_DATASET.DATASET_PATH.SEMANTIC = "semantic_masks"
    cfg.TREES_DATASET.DATASET_PATH.SEMANTIC_ANOMALY = "anomaly_gt"
    cfg.TREES_DATASET.DATASET_PATH.SEMANTIC_BARK = "bark_gt"


    # Solver
    cfg.SOLVER.NAME = "SGD"
    cfg.SOLVER.ACCUMULATE_GRAD = 1
    cfg.SOLVER.FAST_DEV_RUN = None
    cfg.SOLVER.BASE_LR = 0.0013182567385564075
    cfg.SOLVER.N_EPOCHS = 700
    cfg.SOLVER.INIT_WEIGHTS = True
    # Runner
    cfg.BATCH_SIZE = 2
    cfg.CHECKPOINT_PATH_TRAINING = ""
    cfg.CHECKPOINT_PATH_INFERENCE = ""
    cfg.PRECISION = 32
    # Callbacks
    cfg.CALLBACKS = CfgNode()
    cfg.CALLBACKS.CHECKPOINT_DIR = None
