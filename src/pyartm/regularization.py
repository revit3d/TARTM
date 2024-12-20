import torch

class Regularization():
    """
    available regularizations: 'decorrelation'
    supports custom regularizations
    """
    def __init__(self, regularization_params=None):
        #dict showing if the regularization is being used
        self.__regularization_using = { }
        #dict of regularization's parameters
        self.__regularization_params = None
        #dict of regularization's loss function
        self.__regularization_functions = None

        if regularization_params is None:
            self.__regularization_params = { }
        else:
            self.__regularization_params = regularization_params

        self.__regularization_functions = {
            'decorrelation': self.__decorrelation_regularization
        }

        self.__regularization_using = { }
        for reg_name in self.__regularization_functions:
            self.__regularization_using[reg_name] = reg_name in self.__regularization_params.keys()

    def add_custom_regularization(self, regularization: callable, reg_name=None, params=None):
        """
        add a custom regularization
        """
        if reg_name is None:
            reg_name = regularization.__name__

        self.__regularization_using[reg_name] = True
        self.__regularization_functions[reg_name] = regularization
        self.__regularization_params[reg_name] = params

    def switch_regularization(self, reg_name, how='off'):
        if how == 'off':
            self.__regularization_using[reg_name] = False
        elif how == 'on':
            self.__regularization_using[reg_name] = True

    def forward(self, phi, theta):
        """
        calculate the regularization loss
        """
        loss = torch.tensor(0.0, requires_grad=True)

        for reg_name, reg_is_used in self.__regularization_using.items():
            if reg_is_used:
                params = self.__regularization_params[reg_name]
                loss = loss + self.__regularization_functions[reg_name](phi, theta, params)

        return loss

    def __decorrelation_regularization(self, phi, theta, params):
        tau = params['tau']
        loss = torch.tensor(0.0, requires_grad=True)
        phi_T = phi.transpose(0, 1)

        for i in range(len(phi_T)):
            for j in range(len(phi_T)):
                if i != j:
                    loss = loss + torch.sum(phi_T[i] * phi_T[j])
        
        return -tau * loss

    def __sparse_topics_regularization(self, phi, theta, params):
        tau = params['tau']
        loss = torch.tensor(0.0, requires_grad=True)

        #regularization calc here
        
        return -tau * loss