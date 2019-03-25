
============================================================
Deep learning functions for classification and regression
============================================================
  Activations supportd: relu, leakyrelu, tanh, sigmoid, softmax
  Regularizations supported: dropout, L2
  Optimizers supported: gradient descent, momentum, adam
  Cost functions supported: binary cross entropy cost (binary classification),
                            categorical cross entropy (multiclass classification),
                            mse and rmse (regression)
  Other features supported: mini batch, learning rate decay 
  Weight initialization: he
  Data sets are expected to be in the format (features,examples)
 ===========================================================

 ===========================================================
   Example 1
       n_layer_learn().demo()
 ===========================================================
   Example 2
       import deeplearn
       X_train, Y_train, X_test, Y_test = <Your data sets>
       d_model = deeplearn.n_layer_learn()
       d_model.add_layers(X_train.shape[0],"none")
       d_model.add_layers(100,"relu",lambd=0.01,dropout_ratio=0.2)
       d_model.add_layers(50,"leakyrelu",lambd=0.01,leaky_alpha=-0.05,dropout_ratio=0.1)
       d_model.add_layers(30,"relu",lambd=0.01,dropout_ratio=0.1)
       d_model.add_layers(20,"relu",lambd=0.1,dropout_ratio=0.1)
       d_model.add_layers(Y_train.shape[0],"sigmoid")
       d_model.train(X_train, Y_train, optimizer="adam",epochs = 400,learning_rate = 0.03, learning_rate_decay=0.003, momentum_beta = 0.9, rmsprop_beta=0.99, batch_size=16, verbose=True, debug=False)
       predicted_values = d_model.predict(X_test)
       deepnet_test_accuracy,_,_,_ = d_model.deepnet_evaluate(X_test, Y_test)
 ====================================================================

 ====================================================================
   Known Issues
       Unstable for higher values of learning rate in case of ADAM and multi-class classification using softmax
       Remedy - try with diffirent/smaller values of learning rate as well as learning rate decay
 
 =====================================================================
