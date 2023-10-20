import pandas as pd
import numpy as np

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from joblib import load
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam



def build_generator(input_dim, output_dim):
    model = Sequential([
        Dense(10, activation='relu', input_dim=input_dim),
        Dense(output_dim, activation='tanh')  # TanH 활성화 함수 사용
    ])
    return model

def build_discriminator(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def Make():
    data_df = pd.read_csv("t_edge.csv")
    rssi_data = data_df[['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6']].values

    G_INPUT_DIM = 100
    RSSI_DIM = 6
    BATCH_SIZE = 64
    EPOCHS = 200
    mf = 6000
    NUM_BATCHES = 6
    BATCH_SAMPLES = mf // NUM_BATCHES

    generator = build_generator(G_INPUT_DIM, RSSI_DIM)
    discriminator = build_discriminator(RSSI_DIM)

    discriminator.compile(optimizer=Adam(0.001), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.9, from_logits=False), metrics=['accuracy'])

    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(G_INPUT_DIM,))
    fake_rssi = generator(gan_input)
    gan_output = discriminator(fake_rssi)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(0.01), loss='binary_crossentropy')

    all_generated_rssi = []

    real_min = rssi_data.min(axis=0)
    real_max = rssi_data.max(axis=0)

    for batch in range(NUM_BATCHES):
        for epoch in range(EPOCHS):
            idx = np.random.randint(0, rssi_data.shape[0], BATCH_SIZE)
            real_rssi = rssi_data[idx]

            noise = np.random.normal(0, 1, (BATCH_SIZE, G_INPUT_DIM))
            fake_rssi = generator.predict(noise)

            real_labels = np.ones((BATCH_SIZE, 1))
            fake_labels = np.zeros((BATCH_SIZE, 1))
            d_loss_real = discriminator.train_on_batch(real_rssi, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_rssi, fake_labels)

            noise = np.random.normal(0, 1, (BATCH_SIZE, G_INPUT_DIM))
            valid_labels = np.ones((BATCH_SIZE, 1))
            g_loss = gan.train_on_batch(noise, valid_labels)

        noise = np.random.normal(0, 1, (BATCH_SAMPLES, G_INPUT_DIM))
        generated_rssi_batch = generator.predict(noise)

        generated_rssi_batch = 0.5 * (generated_rssi_batch + 1)
        generated_rssi_batch = real_min + generated_rssi_batch * (real_max - real_min)

        all_generated_rssi.extend(generated_rssi_batch)

    kalman_model_x = load("mergex.joblib")
    kalman_model_y = load("mergey.joblib")

    predicted_x_fake = kalman_model_x.predict(all_generated_rssi)
    predicted_y_fake = kalman_model_y.predict(all_generated_rssi)

    generated_df = pd.DataFrame(all_generated_rssi, columns=['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6'])
    generated_df.loc[:,'x'] = pd.Series(predicted_x_fake, index=generated_df.index)
    generated_df.loc[:,'y'] = pd.Series(predicted_y_fake, index=generated_df.index)
    filtered_df = generated_df[(generated_df['x'] >= 0.05) & (generated_df['y'] >= 0.05)]
    original_df = pd.read_csv("t_edge_GAN.csv")
    combined_df = pd.concat([original_df,filtered_df], ignore_index=True)
    combined_df.to_csv("t_edge_GAN.csv", index=False)

while(1):
    Make()

# Define the bins for x and y

