# Coherent-neural-inference-of-complex-valued-nonlinear-photonic-dynamics
Nonlinearity is ubiquitous in nature and technology, yet modelling complex nonlinear wave dynamics often relies on computationally intensive numerical solvers. Machine learning offers faster inference, but existing approaches typically predict intensity-based quantities or fixed input-output mappings, limiting reconstruction of the complete coherent wavefield. Here we introduce coherent neural propagation (CNP), a data-driven framework that learns the coupled amplitude–phase evolution of a complex wavefield and recursively predicts its propagation using a single trained model. Using experimentally informed ground-truth data from four nonlinear optical systems, CNP reconstructs temporal and spectral evolution throughout propagation and achieves approximately 40 dB (a factor of 100 times) lower error-vector magnitude than recurrent and complex-valued neural-network benchmarks. Prediction errors remain low over more than 100 recursive steps, and CNP retains predictive capability beyond the input-power range used for training. CNP also reduces computation time for large-scale inference, providing a route towards accurate and efficient data-driven modelling of coherent nonlinear wave dynamics.
# Code
1.Install Pytorch and necessary dependencies
```bash
pip install -r requirements.txt
```
2.The datasets can be obtained 

3.Run load_data.py to generate normalized Train and Test sets

4.Run utils_tool.py to split the Train and Test sets and creat the weight of loss function

5.Run main.py file to train the model
```bash
The model is first trained using the parameters specified in main.py, then fine-tuned for 8 additional epochs with a learning rate of 1e-5 and input dropout set to 0 to obtain the final results.
```
6.Run main_test.py file to test the model


