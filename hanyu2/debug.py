import tensorflow as tf

def enviromentInfo():
    print("RUNING TENSORFLOW VER:",tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices()))
