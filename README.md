This project focuses on designing a Neural Network (NN) that generates a noise-reduced version of an input communication signal and evaluates the system’s efficiency.
The objective is to take a noisy signal, process it using the trained neural network, and output a cleaner signal that preserves the original information while minimizing noise.


**Project Goals**
1.Develop a neural network model for noise reduction in communication signals.
2.Input: Noisy signal.
3.Output: Noise-reduced signal.
4.Evaluate the efficiency of the system using appropriate signal quality metrics.


**Project Structure**

-data_loader.py           # Loads and preprocesses datasets
-dataset_generator.py     # Script to generate noisy-clean dataset pairs
-evaluate.py              # Model evaluation
-generate_dataset.py      # Dataset creation script
-models_cnn.py            # CNN architecture
-models_dae.py            # Denoising Autoencoder architecture
-models_lstm.py           # LSTM-based architecture
-realtime_test.py         # Real-time noise reduction test
-requirements.txt         # Dependencies
-train.py                 # Training script
-train_model.py           # Alternate training workflow
-utils.py                 # Helper functions
