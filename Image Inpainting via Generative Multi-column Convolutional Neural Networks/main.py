import glob
import pathlib

import PIL
import imageio as imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, ELU, UpSampling2D, Concatenate, Input, Flatten, Dropout, BatchNormalization, \
    LeakyReLU, Dense, Layer
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
plt.rcParams["figure.figsize"] = (7, 7)
IMAGE_SHAPE = (32, 32, 3)
LEARNING_RATE = 3e-5

LAM_REC = 0.99
LAM_ADV = 1 - LAM_REC

DROP_SIZE = 0.5

BATCH_SIZE = 16
EPOCHS = 30
MASK_SIZE = (16, 16)
OFFSET = (16, 16)

pathlib.Path('checkpoints').mkdir(exist_ok=True)
pathlib.Path('generated').mkdir(exist_ok=True)

(y_train, _), (y_test, _) = tf.keras.datasets.cifar10.load_data()
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
y_train = (y_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
y_test = (y_test - 127.5) / 127.5  # Normalize the images to [-1, 1]
# cutout
x_train = tfa.image.cutout(y_train, mask_size=MASK_SIZE, offset=OFFSET)
x_test = tfa.image.cutout(y_test, mask_size=MASK_SIZE, offset=OFFSET)

y = x_test[:25]

fig = plt.figure(figsize=(10, 10))
plt.suptitle("Cutout images", fontsize=16)
for i in range(y.shape[0]):
    plt.subplot(5, 5, i + 1)
    plt.imshow((y[i] + 1.) / 2)
    plt.axis('off')

plt.tight_layout()
plt.savefig('generated/image_at_epoch_0000.png')
plt.show()


class Clip(Layer):
    def __init__(self):
        super(Clip, self).__init__()

    def call(self, inputs):
        clipped_inputs = K.clip(inputs, -1, 1)
        return clipped_inputs


def Generator(input_shape=(32, 32, 3)):
    inputs = Input(input_shape)
    # branch 1 with conv 7x7
    eb1 = Conv2D(filters=32, kernel_size=7, strides=(1, 1), padding='same')(inputs)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same')(eb1)

    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same', dilation_rate=(2, 2))(
        eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same', dilation_rate=(4, 4))(
        eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same', dilation_rate=(8, 8))(
        eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same',
                 dilation_rate=(16, 16))(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    eb1 = ELU()(eb1)
    eb1 = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same')(eb1)
    eb1 = ELU()(eb1)

    eb1 = UpSampling2D(size=(4, 4))(eb1)

    # branch 2 with conv 5x5
    eb2 = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same')(inputs)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)

    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', dilation_rate=(2, 2))(
        eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', dilation_rate=(4, 4))(
        eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', dilation_rate=(8, 8))(
        eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same',
                 dilation_rate=(16, 16))(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)

    eb2 = UpSampling2D(size=(2, 2))(eb2)

    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)
    eb2 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb2)
    eb2 = ELU()(eb2)

    eb2 = UpSampling2D(size=(2, 2))(eb2)

    # branch with conv 3x3
    eb3 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same')(inputs)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)

    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', dilation_rate=(2, 2))(
        eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', dilation_rate=(4, 4))(
        eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', dilation_rate=(8, 8))(
        eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                 dilation_rate=(16, 16))(eb3)
    eb3 = ELU()(eb3)

    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)

    eb3 = UpSampling2D(size=(2, 2))(eb3)

    eb3 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)

    eb3 = UpSampling2D(size=(2, 2))(eb3)

    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)
    eb3 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(eb3)
    eb3 = ELU()(eb3)

    decoder = Concatenate(axis=3)([eb1, eb2, eb3])

    decoder = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(decoder)
    decoder = ELU()(decoder)
    decoder = Conv2D(filters=3, kernel_size=3, strides=(1, 1), padding='same')(decoder)

    # linearly norm to (-1, 1)
    decoder = Clip()(decoder)

    model = Model(inputs=inputs, outputs=[decoder], name='generator')
    return model


generator = Generator()
generator.summary()


def Discriminator(input_shape=(32, 32, 3)):
    inputs = Input(input_shape)
    x = Conv2D(64, (3, 3), strides=2, padding="same", name="conv1")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, (3, 3), strides=2, padding="same", name="conv2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, (3, 3), strides=2, padding="same", name="conv3")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, (3, 3), strides=1, padding="same", name="conv4")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="Discriminator")
    return model


discriminator = Discriminator()
discriminator.summary()


class psnr_metric(tf.keras.metrics.Metric):
    def __init__(self, name='psnr', **kwargs):
        super(psnr_metric, self).__init__(**kwargs)
        self.value = self.add_weight('value', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.value.assign_add(tf.reduce_mean(tf.image.psnr(y_true, y_pred, 2)))
        self.count.assign_add(1)

    def reset_state(self):
        self.value.assign(0)
        self.count.assign(0)

    def result(self):
        return self.value / self.count


class ssim_metric(tf.keras.metrics.Metric):
    def __init__(self, name='ssim', **kwargs):
        super(ssim_metric, self).__init__(**kwargs)
        self.value = self.add_weight('value', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.value.assign_add(tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2)))
        self.count.assign_add(1)

    def reset_state(self):
        self.value.assign(0)
        self.count.assign(0)

    def result(self):
        return self.value / self.count


class GAN(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 discriminator,
                 generator,
                 generator_extra_steps=1,
                 discriminator_extra_steps=1,
                 d_training=True,
                 g_training=True, ):
        super(GAN, self).__init__()

        self.discriminator = discriminator
        self.generator = generator

        self.d_steps = discriminator_extra_steps
        self.g_steps = generator_extra_steps

        self.psnr_metric = psnr_metric(name="psnr")
        self.ssim_metric = ssim_metric(name="ssim")
        self.g_training = g_training
        self.d_training = d_training

    def call(self, x):
        return self.generator(x)

    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
        super(GAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_loss = discriminator_loss

    @property
    def metrics(self):
        return [self.psnr_metric, self.ssim_metric]

    def d_train(self, train: bool):
        if train:
            print("Discriminator training mode is enabled")
        else:
            print("Discriminator training mode is disabled")
        self.d_training = train

    def g_train(self, train: bool):
        if train:
            print("Generator training mode is enabled")
        else:
            print("Generator training mode is disabled")
        self.g_training = train

    @tf.function
    def test_step(self, data):
        x_batch, y_batch = None, None
        if isinstance(data, tuple):
            x_batch = data[0]
            y_batch = data[1]

        fake_images = self.generator(x_batch, training=False)
        fake_logits = self.discriminator(fake_images, training=False)
        real_logits = self.discriminator(y_batch, training=False)

        d_loss = self.discriminator_loss(real_logits, fake_logits)
        g_loss = self.generator_loss(fake_logits, y_batch, fake_images)

        self.psnr_metric.update_state(y_batch, fake_images)
        return {"g_loss": g_loss, "d_loss": d_loss, "psnr": self.psnr_metric.result(), }

    @tf.function
    def train_step(self, data):
        x_batch, y_batch = None, None
        if isinstance(data, tuple):
            x_batch = data[0]
            y_batch = data[1]

        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                fake_images = self.generator(x_batch, training=self.g_training)
                fake_logits = self.discriminator(fake_images, training=self.d_training)
                real_logits = self.discriminator(y_batch, training=self.d_training)

                d_loss = self.discriminator_loss(real_logits, fake_logits)

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            if self.d_training:
                self.discriminator_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        for i in range(self.g_steps):
            with tf.GradientTape() as tape:
                generated_images = self.generator(x_batch, training=self.g_training)
                gen_img_logits = self.discriminator(generated_images, training=self.d_training)
                g_loss = self.generator_loss(gen_img_logits, y_batch, generated_images)

            if self.g_training:
                gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        self.psnr_metric.update_state(y_batch, fake_images)
        return {"g_loss": g_loss, "d_loss": d_loss, "psnr": self.psnr_metric.result(), }


cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()


def discriminator_loss(real_preds, fake_preds):
    real_loss = cross_entropy(tf.ones_like(real_preds), real_preds)
    fake_loss = cross_entropy(tf.zeros_like(fake_preds), fake_preds)
    total_loss = real_loss + fake_loss
    return total_loss


def adv_loss(fake_preds):
    return cross_entropy(tf.ones_like(fake_preds), fake_preds)


def rec_loss(y_true, y_pred):
    # return mse(y_true, y_pred)  / 4    # MSE / L2
    return mae(y_true, y_pred) / 2  # MAE / L1


def generator_loss(fake_logits, real_images, fake_images):
    return LAM_REC * rec_loss(real_images, fake_images) + LAM_ADV * adv_loss(fake_logits)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE * 10, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.9)

gan = GAN(discriminator=Discriminator(),
          generator=Generator())

gan.compile(generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer, )
images_to_show = x_test[:25]
print(images_to_show.shape)


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, images_to_show=None):
        self.images_to_show = images_to_show
        print(self.images_to_show.shape)

    def on_epoch_end(self, epoch, logs=None):
        generated = self.model.generator(self.images_to_show, training=False)

        paddings = tf.constant([[0, 0, ], [OFFSET[0] // 2, OFFSET[1] // 2], [OFFSET[0] // 2, OFFSET[1] // 2], [0, 0]])
        reconstructed_images = self.images_to_show + tf.pad(tf.image.central_crop(generated, 0.5), paddings)
        plt.ioff()

        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(f"epoch:{epoch + 1}")
        for i in range(reconstructed_images.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.imshow((reconstructed_images[i] + 1.) / 2)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('generated/image_at_epoch_{:04d}.png'.format(epoch + 1))
        plt.close(fig)


init_epoch = 0
history = gan.fit(x=x_train,
                  y=y_train,
                  validation_data=(x_test, y_test),
                  epochs=EPOCHS,
                  initial_epoch=init_epoch,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  callbacks=[
                      GANMonitor(images_to_show=images_to_show),
                      ModelCheckpoint(
                          filepath=f'./checkpoints/checkpoint',
                          save_weights_only=True,
                          monitor='val_g_loss',
                          mode='min',
                          save_best_only=True)
                  ])
# Restoration
# gan = GAN(discriminator=Discriminator(),
#           generator=Generator())
#
# gan.compile(generator_loss=generator_loss,
#             discriminator_loss=discriminator_loss,
#             generator_optimizer=generator_optimizer,
#             discriminator_optimizer=discriminator_optimizer, )
# gan.load_weights('./checkpoints/checkpoint')

## Plot train and validation curves
g_loss = history.history['g_loss']
val_g_loss = history.history['val_g_loss']

d_loss = history.history['d_loss']
val_d_loss = history.history['val_d_loss']

psnr = history.history['psnr']
val_psnr = history.history['val_psnr']

plt.figure(figsize=(30, 24))
plt.subplot(3, 1, 1)
plt.plot(g_loss, label='g_loss')
plt.plot(val_g_loss, label='val_g_loss')
plt.legend(loc='lower right')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.title('Training and Validation Generator Loss')

plt.subplot(3, 1, 2)
plt.plot(d_loss, label='d_loss')
plt.plot(val_d_loss, label='val_d_loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0, 3])
plt.title('Training and Validation Discriminator Loss')
plt.xlabel('epoch')

plt.subplot(3, 1, 3)
plt.plot(psnr, label='psnr')
plt.plot(val_psnr, label='val_psnr')
plt.legend(loc='upper right')
plt.ylabel('Metric')
plt.ylim([5, 25])
plt.title('Training and Validation Metric')
plt.xlabel('epoch')
plt.savefig('history.png')
plt.close(fig)


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('generated/image_at_epoch_{:04d}.png'.format(epoch_no))


display_image(history.epoch[-1] + 1)

anim_file = 'dcgan.gif'

filenames = glob.glob('generated/image*.png')
filenames = sorted(filenames)
frames = []
for i, filename in enumerate(filenames):
    if i % 1 == 0:
        image = imageio.v2.imread(filename)
        frames.append(image)
imageio.mimsave(anim_file, frames, format='GIF', fps=4)

# import tensorflow_docs.vis.embed as embed
#
# embed.embed_file(anim_file)
