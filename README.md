# Solve Sudoku 
1. Install the dependencies, perform the following, 
  ```bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install python3-venv build-essential python3-dev git-all libgl1-mesa-dev ffmpeg
  ```
2. Create a virtual environment
 ```bash
    python3 -m venv sudoku_env
 ```
3. Activate the environment
```bash
    source sudoku_env/bin/activate
 ```
4. Install the packages
```bash
    python -m pip install --upgrade pip 
    pip install -r requirements.txt
 ```
5. Train the model using the below command, after the training you should see model **digit_classifier.h5** is created. 
```bash
    python train_digit_classifier.py
 ```
debug message
 ```bash
Epoch 1/5
   1/1875 ━━━━━━━━━━━━━━━━━━━━ 17:59 576ms/step - accuracy: 0.3125 - loss: 2.323   4/1875 ━━━━━━━━━━━━━━━━━━━━ 36s 20ms/step - accuracy: 0.2689 - loss: 2.2632  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9187 - loss: 0.2745 - val_accuracy: 0.9823 - val_loss: 0.0523
Epoch 2/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9862 - loss: 0.0444 - val_accuracy: 0.9885 - val_loss: 0.0365
Epoch 3/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9914 - loss: 0.0287 - val_accuracy: 0.9910 - val_loss: 0.0288
Epoch 4/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9940 - loss: 0.0183 - val_accuracy: 0.9900 - val_loss: 0.0281
Epoch 5/5
   1/1875 ━━━━━━━━━━━━━━━━━━━━ 51s 27ms/step - accuracy: 1.0000 - loss: 4.0385e-   8/1875 ━━━━━━━━━━━━━━━━━━━━ 14s 8ms/step - accuracy: 0.9933 - loss: 0.0165   1875/1875 ━━━━━━━━━━━━━━━━━━━━ 11s 6ms/step - accuracy: 0.9961 - loss: 0.0123 - val_accuracy: 0.9898 - val_loss: 0.0318
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Model saved as digit_classifier.h5
 ```
6. Launch the application to solve Sudoku
```bash
    python solve.py
 ```
