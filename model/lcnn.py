import tensorflow as tf
from keras.layers import Activation, Dense, BatchNormalization, MaxPool2D, Lambda, Input, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.initializers import he_normal

from .layers import Maxout


#function that return the stuck of Conv2D and MFM
def MaxOutConv2D(input, dim, kernel_size, strides, padding='same'):
    """MaxOutConv2D

    This is a helper function for LCNN class.
    This function combine Conv2D layer and Mac Feature Mapping function (MFM).
    Makes codes more readable.

    Args:
      input(tf.Tensor): The tensor from a previous layer.
      dim(int): Dimenstion of the Convolutional layer.
      kernel_size(int): Kernel size of Convolutional layer.
      strides(int): Strides for Convolutional layer.
      padding(string): Padding for Convolutional layer, "same" or "valid".

     Returns:
      mfm_out: Outputs after MFM.
       
    Examples:
      conv2d_1 = MaxOutConv2D(input, 64, kernel_size=2, strides=2, padding="same")

    """
    conv_out = Conv2D(dim, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    mfm_out = Maxout(int(dim/2))(conv_out)
    return mfm_out


#function that return the stuck of FC and MFM
def MaxOutDense(x, dim):
    """ MaxOutDense
    
    Almost same as MaxOutConv2D.
    the difference is just Dense layer but not convolutional layer.

    """
    dense_out = Dense(dim)(x)
    mfm_out = Maxout(int(dim/2))(dense_out)
    return mfm_out

# this function helps to build LCNN. 
def build_lcnn(shape, n_label=2):
    """

    Define LCNN model by using Keras layers

    Augs:
     shape (list) : Input shape for LCNN. (Example : [128, 128, 1])
     n_label (int) : Number of label that LCNN should predict.

    Returns:
      Model (keras.model): LCNN model 

    """
    
    input = Input(shape=shape)

    conv2d_1 = MaxOutConv2D(input, 64, kernel_size=5, strides=1, padding='same')
    maxpool_1 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_1)

    conv_2d_2 = MaxOutConv2D(maxpool_1, 64, kernel_size=1, strides=1, padding='same')
    batch_norm_2 = BatchNormalization()(conv_2d_2)

    conv2d_3 = MaxOutConv2D(batch_norm_2, 96, kernel_size=3, strides=1, padding='same')
    maxpool_3 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_3)
    batch_norm_3 = BatchNormalization()(maxpool_3)

    conv_2d_4 = MaxOutConv2D(batch_norm_3, 96, kernel_size=1, strides=1, padding='same')
    batch_norm_4 = BatchNormalization()(conv_2d_4)

    conv2d_5 = MaxOutConv2D(batch_norm_4, 128, kernel_size=3, strides=1, padding='same')
    maxpool_5 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_5)

    conv_2d_6 = MaxOutConv2D(maxpool_5, 128, kernel_size=1, strides=1, padding='same')
    batch_norm_6 = BatchNormalization()(conv_2d_6)

    conv_2d_7 = MaxOutConv2D(batch_norm_6, 64, kernel_size=3, strides=1, padding='same')
    batch_norm_7 = BatchNormalization()(conv_2d_7)

    conv_2d_8 = MaxOutConv2D(batch_norm_7, 64, kernel_size=1, strides=1, padding='same')
    batch_norm_8 = BatchNormalization()(conv_2d_8)

    conv_2d_9 = MaxOutConv2D(batch_norm_8, 64, kernel_size=3, strides=1, padding='same')
    maxpool_9 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv_2d_9)
    flatten = Flatten()(maxpool_9)

    dense_10 = MaxOutDense(flatten, 160)
    batch_norm_10 = BatchNormalization()(dense_10)
    dropout_10 = Dropout(0.75)(batch_norm_10)

    output = Dense(n_label, activation='softmax')(dropout_10)
            
    return Model(inputs=input, outputs=output)



class LCNN:
    def __init__(self):
        self.model_type = 'LCNN'

    #function that return the stuck of Conv2D and MFM
    def MaxOutConv2D(self, x, dim, kernel_size, strides, padding='same'):
        conv_out = Conv2D(dim, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        mfm_out = Maxout(int(dim/2))(conv_out)
        return mfm_out


    #function that return the stuck of FC and MFM
    def MaxOutDense(self, x, dim):
        dense_out = Dense(dim)(x)
        mfm_out = Maxout(int(dim/2))(dense_out)
        return mfm_out

    
    #this function to save score for ASVspoof2019 is called  after precicting 
    def save_score(self, score, df, path_to_txt):
        cm_score = df[['utt_id', 'attacks', 'key']] 
        cm_score['score'] = score
        cm_score.to_csv(path_to_txt, sep=' ', index=False, header=False)


    def build(self, shape):

        input = Input(shape=shape)

        conv2d_1 = MaxOutConv2D(input, 64, kernel_size=5, strides=1, padding='same')
        maxpool_1 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_1)

        conv_2d_2 = MaxOutConv2D(maxpool_1, 64, kernel_size=1, strides=1, padding='same')
        batch_norm_2 = BatchNormalization()(conv_2d_2)

        conv2d_3 = MaxOutConv2D(batch_norm_2, 96, kernel_size=3, strides=1, padding='same')
        maxpool_3 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_3)
        batch_norm_3 = BatchNormalization()(maxpool_3)

        conv_2d_4 = MaxOutConv2D(batch_norm_3, 96, kernel_size=1, strides=1, padding='same')
        batch_norm_4 = BatchNormalization()(conv_2d_4)

        conv2d_5 = MaxOutConv2D(batch_norm_4, 128, kernel_size=3, strides=1, padding='same')
        maxpool_5 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_5)

        conv_2d_6 = MaxOutConv2D(maxpool_5, 128, kernel_size=1, strides=1, padding='same')
        batch_norm_6 = BatchNormalization()(conv_2d_6)

        conv_2d_7 = MaxOutConv2D(batch_norm_6, 64, kernel_size=3, strides=1, padding='same')
        batch_norm_7 = BatchNormalization()(conv_2d_7)

        conv_2d_8 = MaxOutConv2D(batch_norm_7, 64, kernel_size=1, strides=1, padding='same')
        batch_norm_8 = BatchNormalization()(conv_2d_8)

        conv_2d_9 = MaxOutConv2D(batch_norm_8, 64, kernel_size=3, strides=1, padding='same')
        maxpool_9 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv_2d_9)
        flatten = Flatten()(maxpool_9)

        dense_10 = MaxOutDense(flatten, 160)
        batch_norm_10 = BatchNormalization()(dense_10)
        dropout_10 = Dropout(0.75)(batch_norm_10)

        output = Dense(2, activation='softmax')(dropout_10)
            
        self.model = Model(inputs=input, outputs=output)
        self.model.summary()

    
    def train(self, x_train, y_train, save_path, log_path, validation_data=False,
              lr=0.0001, epochs=50, batch_size=64, earlystopping=10, loss='sparse_categorical_crossentropy'):

        #the path to save trained model
        self.save_path = save_path
        self.lr = lr
        self.batchsize = batch_size

        self.build(x_train.shape[1:])
        self.model.compile(loss = loss,optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

        #callbacks
        es = EarlyStopping(monitor='val_loss', patience=earlystopping , verbose=1)
        cp_cb = ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        print("training lcnn model...")

        if validation_data:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                    shuffle=True, validation_data=validation_data, callbacks=[es, cp_cb])
        else:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                    shuffle=True, validation_split=0.2, callbacks=[es, cp_cb])

        #saving the training log
        df_history = pd.DataFrame(history.history)
        df_history.to_csv(log_path, index=False)
        self.num_epoch = len(df_history)

        print('Done!')

    
    def predict(self, data, df, path_to_score, feature_type=''):
        #if same score exists, remove it 
        print('predicting...')
        if os.path.isfile(path_to_score) == True:
            os.remove(path_to_score)

        #load the model we trained on training phase
        lcnn = load_model(self.save_path, custom_objects={'Maxout': Maxout})
        pred = lcnn.predict(data)
        score = pred[:,[0]] - pred[:,[1]]

        label = np.zeros(len(pred))
        label[df['key'] == 'spoof'] = 1

        eer = calculate_eer(label, score)
        print(f'EER : {eer * 100} %')
        record_eer(self.save_path, feature_type, self.batchsize, self.lr, self.num_epoch, eer*100, self.model_type)

        self.save_score(score, df, path_to_score)


