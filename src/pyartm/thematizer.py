import torch
import pandas as pd

from pyartm.vectorizer import TfidfVectorizer
from pyartm.regularization import Regularization


class Thematizer():
    def __init__(self,
                 data: pd.DataFrame,
                 vectorizer=TfidfVectorizer,
                 vocab=None,
                 n_topics=10,
                 cache_perplexity=False,
                 device='cpu',
                 seed=None):
        if isinstance(seed, int):
            torch.manual_seed(seed)
        if vocab is None:
            self.__pp_data = self.__preprocess_data(data)
        else:
            self.__pp_data = data
            self.__vocab = vocab

        self.__vectorizer = vectorizer
        self.__topic_count = n_topics
        self.__device = device

        self.__cache_perplexity = cache_perplexity
        if cache_perplexity:
            self.perplexity_list = []

    def __preprocess_data(self, data) -> torch.Tensor:
        """preprocess and transform data using given vectorizer"""
        # load data if necessary
        if 'text' not in data.columns:
            if 'path' not in data.columns:
                raise ValueError('data has to contain either \'text\' or \'path\' column')
            data['text'] = data.path.apply(self.__load_document)

        self.__pp_data = self.__vectorizer.fit_transform(data.text)
        self.__vocab = self.__vectorizer.vocab_
        return self.__pp_data

    def __load_document(self, path: str) -> str:
        with open(path, 'r') as file:
            return ''.join(file.readlines())
        
    def NLLLoss(self, regularization):
        eps = 1e-9
        output = torch.matmul(self.__phi, self.__theta)
        regularization_loss = -regularization.forward(self.__phi, self.__theta)
        loss = -torch.sparse.sum(self.__pp_data * Log.apply(output + eps))
        loss = loss + regularization_loss
        return loss
    
    def perplexity(self, loss=None):
        if loss is None:
            loss = self.NLLLoss(regularization=None)
        loss /= torch.sparse.sum(self.__pp_data)
        return torch.exp(loss).cpu().item()

    def fit(self, n_epochs_max=10, regularization=None, online=False, batch_size=1000, alpha=0.7, verbose=True):
        if regularization is None:
            regularization = Regularization()

        if online:
            self.__fit_online(n_epochs_max=n_epochs_max, regularization=regularization, batch_size=batch_size, alpha=alpha, verbose=verbose)
        else:
            self.__fit_offline(n_epochs_max=n_epochs_max, regularization=regularization, verbose=verbose)

    def __fit_step(self, regularization):
        """perform a fit step"""
        # forward
        loss = self.NLLLoss(regularization=regularization)

        # backward
        loss.backward()
        self.__optimizer_step()
        return loss.detach()

    def __fit_offline(self, n_epochs_max, regularization, verbose):
        if self.__pp_data is None:
            self.__pp_data = self.__preprocess_data(self.__data).to(self.__device)
        self.__word_count, self.__doc_count = self.__pp_data.shape

        # initialize state
        self.__phi = torch.rand(self.__word_count, self.__topic_count)
        self.__phi = self.__norm(self.__phi).to(self.__device).requires_grad_(True)
        self.__theta = torch.full((self.__topic_count, self.__doc_count), 1 / self.__topic_count, device=self.__device, requires_grad=True)

        # fit
        for epoch in range(n_epochs_max):
            # step
            loss = self.__fit_step(regularization=regularization)
            perplexity = self.perplexity(loss)
            if self.__cache_perplexity:
                self.perplexity_list.append(perplexity)

            # model quality track
            if verbose:
                print('Epoch [{}/{}], Perplexity: {:.2f}'.format(epoch + 1, n_epochs_max, self.perplexity_list[-1]))

        del self.__pp_data

    def __fit_online(self, n_epochs_max, regularization, batch_size, alpha, verbose):
        # initialize state
        self.__phi = torch.rand(0, self.__topic_count, device=self.__device)

        # iterating through the corpus by batches
        for start in range(0, len(self.__data), batch_size):
            cur_batch_size = min(batch_size, len(self.__data) - start)

            # process and transform batch
            data = self.__data[start:start + cur_batch_size].reset_index()
            self.__pp_data = self.__preprocess_data(data).to(self.__device)

            # initialize hyperparameters
            self.__word_count, self.__doc_count = self.__pp_data.shape

            # update phi if new words were found in preprocessed data
            self.__phi = torch.cat((self.__phi, torch.rand(self.__word_count - self.__phi.shape[0],
                                                           self.__topic_count,
                                                           device=self.__device)))
            self.__phi = self.__norm(self.__phi)

            # initialize theta matrix
            self.__theta = torch.full((self.__topic_count, self.__doc_count),
                                      1 / self.__topic_count,
                                      device=self.__device,
                                      requires_grad=True)

            # fit on a batch
            mem_phi = self.__phi.clone()
            self.__phi.requires_grad = True
            for _ in range(n_epochs_max):
                loss = self.__fit_step(regularization=regularization)

            perplexity = self.perplexity(loss)
            if self.__cache_perplexity:
                self.perplexity_list.append(perplexity)
            self.__phi = self.__phi.detach()

            # update phi
            self.__phi = alpha * mem_phi + (1 - alpha) * self.__phi

            # model quality track
            if verbose:
                print('Batch [{}/{}], Perplexity: {:.2f}'.format(
                    start // batch_size + 1, 
                    (len(self.__data) + (batch_size - 1)) // batch_size,
                    self.perplexity_list[-1]))

            del self.__pp_data
            del self.__theta

    def __norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        implementation of the norm function
        """
        x = torch.maximum(x, torch.tensor(0.0))
        x = x.transpose(0, 1)

        for i in range(len(x)):
            s = torch.sum(x[i])
            if s.data != 0:
                x[i] /= s

        return x.transpose(0, 1)

    def __optimizer_step(self):
        """
        perform an optimization using the norm function
        """
        params = [self.__phi, self.__theta]
        optimized_params = []

        for param in params:
            grad = param.grad.data
            param = self.__norm(param * -grad)
            param = param.detach().requires_grad_(True)
            optimized_params.append(param)

        del self.__phi
        del self.__theta
        self.__phi, self.__theta = optimized_params

    def get_topic_cores(self, n_words=3) -> dict:
        """
        get the most relevant words for all themes with their relevance percentage
        """
        inv_vocab = { index: word for word, index in self.__vocab.items() }
        theme_cores = { }

        for i, topic in enumerate(self.__phi.transpose(0, 1)):
            probs, indices = topic.topk(n_words)
            theme_cores[i] = { inv_vocab[idx.item()]: round(prob.item() * 100, 2) for prob, idx in zip(*[probs, indices]) }

        return theme_cores


class Log(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.log(x)

    @staticmethod
    def backward(ctx, gO):
        x, = ctx.saved_tensors
        #  we cannot do gO / x because sparse division requires
        #  the divisor to be a scalar
        return gO * (1 / x)
