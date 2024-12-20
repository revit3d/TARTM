import torch

from pyartm.thematizer import Thematizer


class em_thematizer(Thematizer):
    def fit_em(self, n_epochs, regularization, verbose=True):
        phi = self.__phi.detach()
        theta = self.__theta.detach()
        data = self.__pp_data.to_dense()

        for epoch in range(n_epochs):
            div = data / torch.matmul(phi, theta)
            new_phi = self.__norm(phi * torch.matmul(div, theta.transpose(0, 1)))
            new_theta = self.__norm(theta * torch.matmul(phi.transpose(0, 1), div))
            phi = new_phi
            theta = new_theta

            #perplexity evaluation
            output = torch.matmul(phi, theta)
            loss = -torch.sum(data * torch.log(output))
            loss /= torch.sum(data)
            loss = torch.exp(loss).cpu().item()

            if self.__cache_perplexity:
                self.perplexity_list.append(loss)

            #model quality track
            if verbose:
                print('Epoch [{}/{}], Perplexity: {:.2f}'.format(epoch + 1, n_epochs, loss))
