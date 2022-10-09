
import dataset as dataset
import tensorflow as tf
from model import Model
import result as result
import config as CONF
import os


class Train:
    def __init__(self,datasets):
        # os.environ['CUDA_VISBIBLE_DEVICES'] = '-1'
        self.dataset = None
        for set in datasets:
            self.train(set)

    def buildDataset(self,dataset_name):
        data,n_classes = dataset.load(dataset_name)
        return data,n_classes

    def buildModel(self,n_classes):
        # Get frontend
        # model_frontend = Model(type="frontend",num_classes=n_classes)
        # model_frontend = model_frontend.getModel()

        # Get backend
        model_backend = Model(type="backend",num_classes=n_classes)
        model_backend = model_backend.getModel()
        return model_backend

    def trainModel(self,model,data):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        callback_logger = tf.keras.callbacks.CSVLogger(
            "./output/traning_log_{}.csv".format(self.dataset), separator=',', append=False
        )
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoint/{}_model.ckpt".format(self.dataset),
                                                         save_weights_only=True,
                                                         verbose=1)

        train = model.fit(data['train'],
                  validation_data=data['valid'],
                  batch_size=CONF.BATCH_SIZE,
                  epochs=CONF.NUM_EPOCHS,
                  steps_per_epoch=CONF.STEPS_PER_EPOCHS,
                  callbacks=[callback_logger,cp_callback])

        return train,model

    def saveResult(self,train):
        output = result.generateResultFromTraining(train)
        path = CONF.TRAIN_LOG_PATH+"fixed_PCEN_pitch_{}.csv".format(self.dataset)
        output.to_csv(path,index_label="epochs")
        print("Training log saved to",path)

    def train(self,set):
        self.dataset = set
        print("✨Start training For {} dataset!".format(set))
        # 1. build dataset
        data,n_classes = self.buildDataset(set)
        # 2. build model
        leaf = self.buildModel(n_classes)
        # 3. training
        train,model = self.trainModel(leaf,data)
        # 4 save model & results
        self.saveResult(train)
        # 5. save the model
        # Display the model's architecture
        model.summary()
        # Save the entire model as a SavedModel.
        model.save(CONF.MODEL_DIR)
        print("{} training finish.".format(set))




