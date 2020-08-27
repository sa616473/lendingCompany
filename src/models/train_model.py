import os

import pandas as pd
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

import kerastuner as kt

import pickle

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard


#1 Model XGBregressor

#We are going to try out differnet n_estimators and learning
#rates to find the best hyperparameters
def xgb_model(n_estimators=[], learning_rate=[], validation_data=(), training_data=(), testing_data=(), directory='', filename=''):
    '''
    Takes a list of estimators and learning rate
    along with train/valid/test data.
    
    Runs the XGB regressor saves
    the weights in .model format and
    the performances in a csv file 
    and returns the performance
    results back in a dataFrame
    '''
    mse = {}
    i = 0
    for estimator in n_estimators:
        for rate in learning_rate:
            #Inisiating the model
            model = XGBClassifier(n_estimators=estimator, 
                                 learning_rate=rate)
            
            #Training the model
            model.fit(training_data[0], training_data[1],
                     early_stopping_rounds=50,
                     eval_set = [(validation_data[0], validation_data[1])],
                     verbose=False)
            
                        #saving the model
            model.save_model('../src/models/xgb_weights/n_estimator{}_learning_rate{}.model'  
                             .format(estimator, rate))
            #Evaluating the model
            prediction = model.predict(testing_data[0])
            
            #Calculating the error
            accuracy = accuracy_score(prediction, testing_data[1].values)
            mse[i] = [estimator, rate, accuracy]
            i = i + 1
    
    #Converting the dict to a DataFrame
    xgb_performance = pd.DataFrame(data=mse)
    xgb_performance = xgb_performance.transpose()
    xgb_performance.columns = ['n_estimator', 'learning_rate', 'accuracy']
    xgb_performance.index.name = 'index'
    
    #Saving the performances in a CSV file
    if os.path.exists(directory):
        xgb_performance.to_csv('../src/models/{}/{}.csv'.format(directory,filename))
    else:
        os.makedirs(directory)
        xgb_performance.to_csv('../src/models/{}/{}.csv'.format(directory,filename))
    return xgb_performance

#2nd model neural networks
def model_build(hp):
    model = Sequential()
    input_shape = (78,)
    
    #Activation function and neural units to choose from
    hp_units_1 = hp.Int('units', min_value = 4,max_value = 78,step = 4, default=16)
    hp_units_2 = hp.Int('units', min_value = 4,max_value = 78,step = 4, default=16)
    hp_units_3 = hp.Int('units', min_value = 4,max_value = 78,step = 4, default=16)

    
    hp_choice_1 = hp.Choice('dense_activation',
                          values=['relu','elu', 'sigmoid'],
                            default='relu')
    hp_choice_2 = hp.Choice('dense_activation',
                          values=['relu','elu', 'sigmoid'],
                            default='relu')
    hp_choice_3 = hp.Choice('dense_activation',
                          values=['relu','elu', 'sigmoid'],
                            default='relu')
    #For Dropout layer
    hp_float_1 = hp.Float('dropout',
                       min_value=0.0,
                       max_value=0.5,
                       default=0.15,
                       step=0.01)
    
    hp_float_2 = hp.Float('dropout',
                       min_value=0.0,
                       max_value=0.5,
                       default=0.15,
                       step=0.01)
    hp_float_3 = hp.Float('dropout',
                       min_value=0.0,
                       max_value=0.5,
                       default=0.15,
                       step=0.01)

    model.add(Dense(units= hp_units_1,
                   activation =hp_choice_1,
                    input_shape = input_shape))
    
    model.add(Dropout(hp_float_1))

    model.add(Dense(units= hp_units_2,
                   activation =hp_choice_2))

    model.add(Dropout(hp_float_2))
    
    model.add(Dense(units= hp_units_3,
                   activation =hp_choice_3))
    
    model.add(Dropout(hp_float_3))

    model.add(Dense(1, activation='sigmoid'))

    #Learning rate
    hp_learning_rate = hp.Choice('learning_rate', values= [0.01, 0.001, 0.001])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                 loss = 'binary_crossentropy',
                 metrics=['accuracy'])

    return model

#For Display
import IPython
class ClearTrainingOutput(Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


def hyper_parameter_search(search_type='BO',objective='accuracy', seed=101, 
                           max_trails=10, directory=os.path.normpath('C:/'), 
                           project_name='', max_epochs=10, factor=3, epochs=10,
                          train_data=(), val_data=()):
    '''
    Given the search type this method uses that optimization 
    method from keras tuner and finds the best parameters.
    and returns the model with the best parameteres. 
    '''
    search_type = search_type.upper()
    
    if search_type == 'BO' or search_type == 'BAYESIANOPTIMIZATION':
        tuner = kt.BayesianOptimization(model_build,
                                  objective=objective,
                                        seed=seed,
                                  max_trials=max_trails,
                                   directory=directory,
                                  project_name=project_name)
    
    elif search_type == 'RS' or search_type == 'RANDOMSEARCH':
        tuner = kt.RandomSearch(model_build,
                          objective=objective,
                          seed=seed,
                          max_trials=max_trails,
                          directory=directory,
                            project_name = project_name)
    
    elif search_type == 'HB' or search_type == 'HYPERBAND':
        tuner = kt.Hyperband(model_build,
                     max_epochs=max_epochs,
                       objective=objective,
                       factor=factor,
                     directory=directory,
                    project_name=project_name)
    else:
        raise ValueError('The requested keras tuner search type doesnot exist\n')
    
    tuner.search(train_data[0], train_data[1], epochs=epochs, 
               validation_data = (val_data[0], val_data[1]),
               callbacks = [ClearTrainingOutput()], verbose=1)
    
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    
    print(f"""
        The hyperparameter search is complete. The optimal units
        {best_hps.get('units')} and the optimal learning rate is 
        {best_hps.get('learning_rate')} and the optimal dropout
        {best_hps.get('dropout')} and the optimal activation
        {best_hps.get('dense_activation')}.""")
    model = tuner.hypermodel.build(best_hps)
    return model


def model_fit( model, train_data =(),val_data=(), monitor='accuracy', patience=3, verbose=2, epochs=50):
    '''
    Fits the model to the validation data.
    '''
    log_directory = 'tensorboard\\fit'
    
    board = TensorBoard(log_dir=log_directory,histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1)
    
    callback = EarlyStopping(monitor=monitor, patience=patience, verbose=verbose)
    history = model.fit(train_data[0], train_data[1], epochs=epochs,
                       validation_data = (val_data[0], val_data[1]),
                      callbacks = [callback, board], verbose=1,batch_size=128)
    return history, model