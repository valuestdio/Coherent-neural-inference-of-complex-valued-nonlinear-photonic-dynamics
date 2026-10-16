# Coherent-neural-inference-of-complex-valued-nonlinear-photonic-dynamics
# Code
1.Install Pytorch and necessary dependencies
```bash
pip install -r requirements.txt
```
2.The datasets can be obtained 

3.Run load_data.py to generate normalized Train and Test sets

4.Run utils_tool.py to split the Train and Test sets and creat the weight of loss function

5.Run main.py file to train the model
The model is first trained using the parameters specified in main.py, then fine-tuned for 8 additional epochs with a learning rate of 1e-5 and input dropout set to 0 to obtain the final results.

6.Run main_test.py file to test the model


