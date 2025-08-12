This project focuses on designing a Neural Network (NN) that generates a noise-reduced version of an input communication signal and evaluates the system’s efficiency.
The objective is to take a noisy signal, process it using the trained neural network, and output a cleaner signal that preserves the original information while minimizing noise.


**Project Goals**
1.Develop a neural network model for noise reduction in communication signals.<br>
2.Input: Noisy signal.<br>
3.Output: Noise-reduced signal.<br>
4.Evaluate the efficiency of the system using appropriate signal quality metrics.<br>


**Project Structure**

-data_loader.py           # Loads and preprocesses datasets<br>
-dataset_generator.py     # Script to generate noisy-clean dataset pairs<br>
-evaluate.py              # Model evaluation<br>
-generate_dataset.py      # Dataset creation script<br>
-models_cnn.py            # CNN architecture<br>
-models_dae.py            # Denoising Autoencoder architecture<br>
-models_lstm.py           # LSTM-based architecture<br>
-realtime_test.py         # Real-time noise reduction test<br>
-requirements.txt         # Dependencies<br>
-train.py                 # Training script<br>
-train_model.py           # Alternate training workflow<br>
-utils.py                 # Helper functions<br>
