import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        del_w_input_hidden = np.zeros(self.weights_input_to_hidden.shape)
        del_w_hidden_output = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
        # Implement the forward pass function below
        # Implement the backproagation function below
            del_w_input_hidden, del_w_hidden_output = self.backpropagation(final_outputs, hidden_outputs, X, y, del_w_input_hidden, del_w_hidden_output)
        
        self.update_weights(del_w_input_hidden, del_w_hidden_output, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_output, X, y, del_w_input_hidden, del_w_hidden_output):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            del_w_input_hidden: change in weights from input to hidden layers
            del_w_hidden_output: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        ## Backward pass ##
#         # TODO: Calculate the network's prediction error
#         error = y - output

#         # TODO: Calculate error term for the output unit
#         output_error_term = error * output * (1 - output)

#         ## propagate errors to hidden layer

#         # TODO: Calculate the hidden layer's contribution to the error
#         hidden_error = np.dot(output_error_term, weights_hidden_output)

#         # TODO: Calculate the error term for the hidden layer
#         hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

#         # TODO: Update the change in weights
#         del_w_hidden_output += output_error_term * hidden_output
#         del_w_input_hidden += hidden_error_term * x[:, None]
        
        # TODO: Output error
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        output_error_term = error
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        # output_error_term = error * final_outputs * (1 - final_outputs)
        
         # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)
        
         # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output[:,None]
        del_w_input_hidden += hidden_error_term * X[:, None]

        
        return del_w_input_hidden, del_w_hidden_output
    
    def update_weights(self, del_w_input_hidden, del_w_hidden_output, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            del_w_input_hidden: change in weights from input to hidden layers
            del_w_hidden_output: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * del_w_hidden_output / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * del_w_input_hidden / n_records # update input-to-hidden weights with gradient descent step
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################

