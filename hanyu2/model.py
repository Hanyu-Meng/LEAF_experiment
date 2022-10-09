from network import models
import config as CONF
import tensorflow as tf

# ↓↓↓ 这里你可以自由修改你前后端对应模型
MODEL_TYPE = {
    "backend" : models.AudioClassifier,
    # "frontend": tf.keras.models.load_model('../saved_model/LEAF'),
}

class Model:
    def __init__(self,num_classes = None,type="frontend"):
        self.model = self.prepareModel(type,num_classes)

    def prepareModel(self,type,num_classes = None):
        # Check if model type is vaild
        if type in MODEL_TYPE:
            if type == "backend":
                # Remainder **kwargs need to handle!!!!
                model = MODEL_TYPE[type](num_classes)
                # Compile model
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                model.compile(loss=loss_fn,
                              optimizer=tf.keras.optimizers.Adam(CONF.BACKEND_LEARNING_RATE),
                              metrics=[CONF.METRIC])
                return model

            elif type == "frontend":
                model = MODEL_TYPE[type]
                return model
        raise Exception("模型类型不正确！只能: frontend | backend ")

    def getModel(self):
        return self.model